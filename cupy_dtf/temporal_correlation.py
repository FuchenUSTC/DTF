import torch
from torch.autograd import Function
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from torch import Tensor
from utils import Dtype, Stream, load_kernel

CUDA_NUM_THREADS = 1024

kernel_loop = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)
'''


def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS


# NCTHW -> correlation between adjacent frames
_correlation_forward_kernel = kernel_loop + '''
extern "C"
__global__ void correlation_forward_kernel(
const ${Dtype}* bottom_data, const ${Dtype}* weight_data, ${Dtype}* top_data) {
  CUDA_KERNEL_LOOP(index, ${out_nthreads}) {
    const int c_per_group = ${kernel_w} * ${kernel_h};
    const int c_bottom_per_group = ${bottom_channels} / ${group_num};  // the group size
    const int n = (index / ${top_channels} / ${bottom_times} / ${bottom_height} / ${bottom_width}); 
    const int c = (index / ${bottom_times} / ${bottom_height} / ${bottom_width}) % ${top_channels};
    const int t = (index / ${bottom_height} / ${bottom_width}) % ${bottom_times};
    const int h = (index / ${bottom_width}) % ${bottom_height};
    const int w = (index % ${bottom_width});
    const int group_id = c / c_per_group;
    const int c_id = c % (c_per_group);
    

    int offset_former_time = t - 1;
    if (t <= 0){
      offset_former_time = 0;
    }

    ${Dtype} value = 0;
    int kh = (c_id / ${kernel_w}) % ${kernel_h};
    int kw = (c_id % ${kernel_w});
    const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
    const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
    if ((h_in >= 0) && (h_in < ${bottom_height}) && (w_in >= 0) && (w_in < ${bottom_width})) {
      for (int cc = group_id * c_bottom_per_group; cc < (group_id+1)*c_bottom_per_group; ++cc){
        const int offset_bottom = (((n * ${bottom_channels} + cc) * ${bottom_times} + t) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
        const int offset_bottom_former = (((n * ${bottom_channels} + cc) * ${bottom_times} + offset_former_time) * ${bottom_height} + h) * ${bottom_width} + w;
        const int offset_weight = (cc * ${bottom_times} + t) * ${kernel_h} * ${kernel_w} + kh * ${kernel_w} + kw;
        value += (weight_data[offset_weight] * bottom_data[offset_bottom_former] * bottom_data[offset_bottom]);
      }
    }
    top_data[index] = value;
  }
}
'''

_correlation_input_backward_kernel = kernel_loop + '''
extern "C"
__global__ void correlation_input_backward_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const weight_data, const ${Dtype}* const bottom_data, ${Dtype}* bottom_diff) {
  CUDA_KERNEL_LOOP(index, ${in_nthreads}) {
    const int c_bottom_per_group = ${bottom_channels} / ${group_num};
    const int n = (index / ${bottom_channels} / ${bottom_times} / ${bottom_height} / ${bottom_width});
    const int c = (index / ${bottom_times} / ${bottom_height} / ${bottom_width}) % ${bottom_channels};
    const int t = (index / ${bottom_height} / ${bottom_width}) % ${bottom_times};
    const int h = (index / ${bottom_width}) % ${bottom_height};
    const int w = (index % ${bottom_width});

    const int group_id = c / c_bottom_per_group;
    
    int offset_former_time = t - 1;
    if (t == 0) {
      offset_former_time = 0;
    }
  
    ${Dtype} value = 0;
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        const int h_out_s = h + ${pad_h} - kh * ${dilation_h};
        const int w_out_s = w + ${pad_w} - kw * ${dilation_w};
        if (((h_out_s % ${stride_h}) == 0) && ((w_out_s % ${stride_w}) == 0)) {
          const int h_out = h_out_s / ${stride_h};
          const int w_out = w_out_s / ${stride_w};
          if ((h_out >= 0) && (h_out < ${top_height}) && (w_out >= 0) && (w_out < ${top_width})) {
            const int offset_top_ch = (group_id *${kernel_h} + kh) * ${kernel_w} + kw;
            const int offset_top = (((n * ${top_channels} + offset_top_ch) * ${bottom_times} + t) * ${top_height} + h_out) * ${top_width} + w_out;
            const int offset_bottom = (((n * ${bottom_channels} + c) * ${bottom_times} + offset_former_time) * ${bottom_height} + h_out) * ${bottom_width} + w_out;
            const int offset_weight = (c * ${bottom_times} + t) * ${kernel_h} * ${kernel_w} + kh * ${kernel_w} + kw; // the weight time offset equals to top offset
            value += weight_data[offset_weight] * top_diff[offset_top] * bottom_data[offset_bottom]; 
          }
        }
      }
    }
    // On t = 0, correlation should be considered twice.
    // Assume the "index" bottom the former, compute the next
    if (t == 0) {
      for (int kh = 0; kh < ${kernel_h}; ++kh) {
        for (int kw = 0; kw < ${kernel_w}; ++kw) {
          const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
          const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
          if ((h_in >= 0) && (h_in < ${bottom_height}) && (w_in >= 0) && (w_in < ${bottom_width})) {
            const int offset_top_ch = (group_id * ${kernel_h} + kh) * ${kernel_w} + kw;
            const int offset_top_next = (((n * ${top_channels} + offset_top_ch) * ${bottom_times}) * ${top_height} + h) * ${top_width} + w;
            const int offset_bottom_next = (((n * ${bottom_channels} + c) * ${bottom_times}) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
            const int offset_weight_next = (c * ${bottom_times}) * ${kernel_h} * ${kernel_w} + kh * ${kernel_w} + kw;
            value +=  weight_data[offset_weight_next] * top_diff[offset_top_next] * bottom_data[offset_bottom_next];
          }
        }
      }
    }
    if (t < ${bottom_times} - 1){
      int offset_next_time = t + 1;
      for (int kh = 0; kh < ${kernel_h}; ++kh) {
        for (int kw = 0; kw < ${kernel_w}; ++kw) {
          const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
          const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
          if ((h_in >= 0) && (h_in < ${bottom_height}) && (w_in >= 0) && (w_in < ${bottom_width})) {
            const int offset_top_ch = (group_id * ${kernel_h} + kh) * ${kernel_w} + kw;
            const int offset_top = (((n * ${top_channels} + offset_top_ch) * ${bottom_times} + offset_next_time) * ${top_height} + h) * ${top_width} + w;
            const int offset_bottom = (((n * ${bottom_channels} + c) * ${bottom_times} + offset_next_time) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
            const int offset_weight = (c * ${bottom_times} + offset_next_time) * ${kernel_h} * ${kernel_w} + kh * ${kernel_w} + kw; // the weight time offset equals to top offset
            value += weight_data[offset_weight] * top_diff[offset_top] * bottom_data[offset_bottom];
          }
        }          
      }
    }
    bottom_diff[index] = value;
  }
}
'''

_correlation_weight_backward_kernel = kernel_loop + '''
extern "C"
__global__ void correlation_weight_backward_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const bottom_data, ${Dtype}* weight_diff) {
  CUDA_KERNEL_LOOP(index, ${wg_nthreads}) {
    const int c_weight_per_group = ${bottom_channels} / ${group_num};
    const int c =  (index / ${bottom_times}  / ${weight_height} / ${weight_width}) % ${bottom_channels};
    const int t =  (index / ${weight_height} / ${weight_width}) % ${bottom_times}; // the same with the top data
    const int kh = (index / ${weight_width}) % ${weight_height};
    const int kw = (index % ${weight_width});
    const int group_id = c / c_weight_per_group; 

    int offset_former_time = t - 1;
    if (t <= 0){
      offset_former_time = 0;
    }

    // scan each position of top data, find the kernel postion and return  
    ${Dtype} value = 0;
    for (int n = 0; n < ${bottom_batch}; ++n){
      for (int h = 0; h < ${top_height}; ++h){
        for (int w = 0; w < ${top_width}; ++w){
          const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
          const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
          if ((h_in >= 0) && (h_in < ${bottom_height}) && (w_in >= 0) && (w_in < ${bottom_width})) {
            const int offset_top_ch = (group_id * ${kernel_h} + kh) * ${kernel_w} + kw ;
            const int offset_top = (((n * ${top_channels} + offset_top_ch) * ${bottom_times} + t) * ${top_height} + h) * ${top_width} + w;
            const int offset_bottom = (((n * ${bottom_channels} + c) * ${bottom_times} + t) * ${bottom_height} + h_in) * ${bottom_width} + w_in;
            const int offset_bottom_former = (((n * ${bottom_channels} + c) * ${bottom_times} + offset_former_time) * ${bottom_height} + h) * ${bottom_width} + w;
            value += bottom_data[offset_bottom] * bottom_data[offset_bottom_former] * top_diff[offset_top];
          }
        }
      }
    }
    weight_diff[index] = value;
  }
}
'''

# NCTHW 3D video correlation
class CorrelationZeropad(Function):
    @staticmethod
    def forward(ctx, input, weight, kernel_size, stride, padding, dilation, group_size):
        kernel_size, stride, padding, dilation, group_size = _pair(kernel_size), _pair(stride), _pair(padding), _pair(dilation), group_size
        ctx.kernel_size, ctx.stride, ctx.padding, ctx.dilation, ctx.group_size = kernel_size, stride, padding, dilation, group_size
        assert input.dim() == 5 and input.is_cuda and weight.is_cuda
        batch_size, input_channels, input_times, input_height, input_width = input.size()
        weight_channels, weight_times, weight_height, weight_width = weight.size()
        output_height = int((input_height + 2 * padding[0] - (dilation[0] * (kernel_size[0] - 1) + 1)) / stride[0] + 1)
        output_width = int((input_width + 2 * padding[1] - (dilation[1] * (kernel_size[1] - 1) + 1)) / stride[1] + 1)
        #assert output_height * output_width == weight_height * weight_width
        group_num = input_channels // group_size
        output_channels = int(weight_height * weight_width) * group_num
        assert output_height * output_width == input_height * input_width
        output = input.new(batch_size, output_channels, input_times, output_height, output_width)
        out_n = output.numel()
        if not input.is_contiguous():
            input = input.detach().clone()
        if not weight.is_contiguous():
            weight = weight.detach().clone()

        with torch.cuda.device_of(input):
            f = load_kernel('correlation_forward_kernel', _correlation_forward_kernel, Dtype=Dtype(input), 
                            out_nthreads = out_n, num=batch_size, weight_channels=weight_channels,
                            bottom_channels=input_channels, bottom_times = input_times, 
                            bottom_height=input_height, bottom_width=input_width,
                            top_height=output_height, top_width=output_width, top_channels = output_channels,
                            kernel_h=kernel_size[0], kernel_w=kernel_size[1],
                            stride_h=stride[0], stride_w=stride[1],
                            dilation_h=dilation[0], dilation_w=dilation[1],
                            pad_h=padding[0], pad_w=padding[1], group_num=group_num)
            f(block=(CUDA_NUM_THREADS, 1, 1),
              grid=(GET_BLOCKS(out_n), 1, 1),
              args=[input.data_ptr(), weight.data_ptr(), output.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        ctx.save_for_backward(input, weight)
        return output

    
    @staticmethod
    def backward(ctx, grad_output):
        kernel_size, stride, padding, dilation, group_size = ctx.kernel_size, ctx.stride, ctx.padding, ctx.dilation, ctx.group_size
        input, weight = ctx.saved_tensors
        assert grad_output.is_cuda
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        batch_size, input_channels, input_times, input_height, input_width = input.size()
        group_num = input_channels // group_size
        weight_channels, weight_times, weight_height, weight_width = weight.size()

        output_height, output_width = grad_output.size()[3:] # NCTHW
        output_channels = int(weight_height * weight_width) * group_num
        grad_input, grad_weight = None, None
        opt = dict(Dtype=Dtype(grad_output),
                   num=batch_size, input_channels=input_channels, input_times=input_times,
                   weight_height = weight_height, weight_width=weight_width,weight_channels=weight_channels,
                   bottom_channels=input_channels, bottom_batch=batch_size,
                   bottom_height=input_height, bottom_width=input_width,
                   bottom_times=input_times, top_channels= output_channels,
                   top_height=output_height, top_width=output_width,
                   kernel_h=kernel_size[0], kernel_w=kernel_size[1],
                   stride_h=stride[0], stride_w=stride[1],
                   dilation_h=dilation[0], dilation_w=dilation[1],
                   pad_h=padding[0], pad_w=padding[1], group_num=group_num)
        in_n = input.numel()
        with torch.cuda.device_of(input):
            if ctx.needs_input_grad[0]:
                grad_input = input.new(input.size())
                n = grad_input.numel()
                opt['in_nthreads'] = n
                f = load_kernel('correlation_input_backward_kernel', _correlation_input_backward_kernel, **opt)
                f(block=(CUDA_NUM_THREADS, 1, 1),
                  grid=(GET_BLOCKS(n), 1, 1),
                  args=[grad_output.data_ptr(), weight.data_ptr(), input.data_ptr(), grad_input.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
            if ctx.needs_input_grad[1]:
                grad_weight = weight.new(weight.size())
                n = grad_weight.numel()
                opt['wg_nthreads'] = n
                f = load_kernel('correlation_weight_backward_kernel', _correlation_weight_backward_kernel, **opt)
                f(block=(CUDA_NUM_THREADS, 1, 1),
                  grid=(GET_BLOCKS(n), 1, 1),
                  args=[grad_output.data_ptr(), input.data_ptr(), grad_weight.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
        return grad_input, grad_weight, None, None, None, None, None


def correlation_zeropad(input, weight, kernel_size=3, stride=1, padding=0, dilation=1, group_size=32):
    assert input.shape[1] == weight.shape[0] and (input.shape[1] % weight.shape[0] == 0)
    if input.is_cuda:
        out = CorrelationZeropad.apply(input, weight, kernel_size, stride, padding, dilation, group_size)
    else:
        out = CorrelationZeropad.apply(input.cuda(), weight.cuda(), kernel_size, stride, padding, dilation, group_size)
        torch.cuda.synchronize()
        out = out.cpu()
    return out


class TemporalCorrelation(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        pad_mode: int = 0,
        group_size: int = 32,
    ):
        super(TemporalCorrelation, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pad_mode = pad_mode
        self.group_size = group_size

    def forward(self, input: Tensor, weight: Tensor):
        out = correlation_zeropad(
            input, 
            weight, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation,
            group_size=self.group_size) 
        return out


def test_correlation_zeropad():
    kernel_size, stride, dilation = 7, 1, 2
    padding = (dilation * (kernel_size - 1) + 1) // 2
    n, c_x, in_time, in_height, in_width = 2, 8, 3, 7, 7
    out_height = int((in_height + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    out_width = int((in_width + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    unfold_test = torch.nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
    w = torch.randn(c_x, in_time, kernel_size, kernel_size, requires_grad=True).double().cuda()
    
    x = torch.randn(n, c_x, in_time, in_height, in_width, requires_grad=True).double().cuda()
    xq = x.view([n, c_x*in_time, in_height, in_width])
    xq_unfold = unfold_test(xq).view(n, c_x, in_time, pow(kernel_size, 2), out_height, out_width)
    
    x_shift = x.clone()
    x_shift[:,:,1:,:,:] = x[:,:,:-1,:,:]
    ww = w.view([1, c_x, in_time, pow(kernel_size,2)]).unsqueeze(4).unsqueeze(5)
    wp = ww.expand([n, c_x, in_time, pow(kernel_size,2), in_height, in_width])
    xs_dup = x_shift.unsqueeze(3).expand([n, c_x, in_time, pow(kernel_size,2), in_height, in_width])
    corre = (xs_dup * xq_unfold * wp).sum(1).permute([0,2,1,3,4])


    y1 = correlation_zeropad(x, w, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, group_size=c_x)
    
    assert (corre - y1).abs().max() < 1e-9
    print('temporal correlation: test forward passed.')
    
    gx1 = torch.autograd.grad(y1.mean(), x, retain_graph=True)[0]
    gx2 = torch.autograd.grad(corre.mean(), x, retain_graph=True)[0]
    #print((gx1-gx2).abs())
    assert (gx1 - gx2).abs().max() < 1e-9
    print('temporal correlation: test backward data passed.')

    gw1 = torch.autograd.grad(y1.mean(), w, retain_graph=True)[0]
    gw2 = torch.autograd.grad(corre.mean(), w, retain_graph=True)[0]
    assert (gw1 - gw2).abs().max() < 1e-9
    print('temporal correlation: test backward weight passed.')

    from functools import partial
    assert torch.autograd.gradcheck(partial(correlation_zeropad, kernel_size=kernel_size, 
                                    stride=stride, padding=padding, dilation=dilation, group_size=c_x), (x, w))
    print('temporal correlation: test case passed.')


def test_correlation_zeropad_1ks():
    kernel_size, stride, dilation = 1, 1, 1
    padding = (dilation * (kernel_size - 1) + 1) // 2
    n, c_x, in_time, in_height, in_width = 2, 8, 3, 7, 7
    out_height = int((in_height + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    out_width = int((in_width + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    unfold_test = torch.nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
    w = torch.randn(c_x, in_time, kernel_size, kernel_size, requires_grad=True).double().cuda()
    
    x = torch.randn(n, c_x, in_time, in_height, in_width, requires_grad=True).double().cuda()
    xq = x.view([n, c_x*in_time, in_height, in_width])
    xq_unfold = unfold_test(xq).view(n, c_x, in_time, pow(kernel_size, 2), out_height, out_width)
    
    x_shift = x.clone()
    x_shift[:,:,1:,:,:] = x[:,:,:-1,:,:]
    ww = w.view([1, c_x, in_time, pow(kernel_size,2)]).unsqueeze(4).unsqueeze(5)
    wp = ww.expand([n, c_x, in_time, pow(kernel_size,2), in_height, in_width])
    xs_dup = x_shift.unsqueeze(3).expand([n, c_x, in_time, pow(kernel_size,2), in_height, in_width])
    corre = (xs_dup * xq_unfold * wp).sum(1).permute([0,2,1,3,4])

    y1 = correlation_zeropad(x, w, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, group_size=c_x)
    
    assert (corre - y1).abs().max() < 1e-9
    print('temporal correlation: test_1ks forward passed.')
    
    gx1 = torch.autograd.grad(y1.mean(), x, retain_graph=True)[0]
    gx2 = torch.autograd.grad(corre.mean(), x, retain_graph=True)[0]
    assert (gx1 - gx2).abs().max() < 1e-9
    print('temporal correlation: test_1ks backward data passed.')

    gw1 = torch.autograd.grad(y1.mean(), w, retain_graph=True)[0]
    gw2 = torch.autograd.grad(corre.mean(), w, retain_graph=True)[0]
    assert (gw1 - gw2).abs().max() < 1e-9
    print('temporal correlation: test_1ks backward weight passed.')

    from functools import partial
    assert torch.autograd.gradcheck(partial(correlation_zeropad, kernel_size=kernel_size, 
                                            stride=stride, padding=padding, dilation=dilation, group_size=c_x), (x, w))
    print('temporal correlation: test_1ks case passed.')


def test_gradcheck():
    kernel_size, stride, dilation = 7, 1, 2
    padding = (dilation * (kernel_size - 1) + 1) // 2
    n, c_x, in_time, in_height, in_width = 2, 16, 3, 7, 7
    group_c = 4
    out_height = int((in_height + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    out_width = int((in_width + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)  
    x = torch.randn(n, c_x, in_time, in_height, in_width, requires_grad=True).double().cuda()
    w = torch.randn(c_x, in_time, kernel_size, kernel_size, requires_grad=True).double().cuda()
    from functools import partial
    y = correlation_zeropad(x, w, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, group_size=group_c)
    assert torch.autograd.gradcheck(partial(correlation_zeropad, kernel_size=kernel_size, 
                                    stride=stride, padding=padding, dilation=dilation, group_size=group_c), (x, w))
    print('temporal correlation: test case passed with group_size=%d.'%(group_c))    


def test_gradcheck_1ks():
    kernel_size, stride, dilation = 1, 1, 1
    padding = (dilation * (kernel_size - 1) + 1) // 2
    n, c_x, in_time, in_height, in_width = 2, 16, 3, 7, 7
    group_c = 4
    out_height = int((in_height + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)
    out_width = int((in_width + 2 * padding - (dilation * (kernel_size - 1) + 1)) / stride + 1)  
    x = torch.randn(n, c_x, in_time, in_height, in_width, requires_grad=True).double().cuda()
    w = torch.randn(c_x, in_time, kernel_size, kernel_size, requires_grad=True).double().cuda()
    from functools import partial
    y = correlation_zeropad(x, w, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, group_size=group_c)
    assert torch.autograd.gradcheck(partial(correlation_zeropad, kernel_size=kernel_size, 
                                    stride=stride, padding=padding, dilation=dilation, group_size=group_c), (x, w))
    print('temporal correlation: test_1ks case passed with group_size=%d.'%(group_c))    


if __name__ == '__main__':
    print('--------- temporal correlation testing ---------')
    test_correlation_zeropad()
    test_correlation_zeropad_1ks()
    test_gradcheck()
    test_gradcheck_1ks()
    print('--------------------- Done ---------------------')