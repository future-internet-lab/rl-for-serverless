import gc

import torch

## MEM utils ##
def mem_report():
    '''Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported'''

    def _mem_report(tensors, mem_type):
        '''f.write the selected tensors of type

        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)

        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation '''
            
        with open("memlog.txt", 'a') as f:    
            f.write('Storage on %s\n' %(mem_type))
            f.write('-'*LEN)
            f.write('\n')
        total_numel = 0
        total_mem = 0
        visited_data = []
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel*element_size /1024/1024 # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())
            with open("memlog.txt", 'a') as f:    
                f.write('%s\t\t%s\t\t%.2f\n' % (
                    element_type,
                    size,
                    mem) )
        with open("memlog.txt", 'a') as f:    
            f.write('-'*LEN)
            f.write("\n")
            f.write('Total Tensors: %d \tUsed Memory Space: %.2f MBytes\n' % (total_numel, total_mem) )
            f.write('-'*LEN)
            f.write("\n")

    LEN = 65
    with open("memlog.txt", 'a') as f:    
        f.write('='*LEN)
        f.write("\n")
        objects = gc.get_objects()
        f.write('%s\t%s\t\t\t%s' %('Element type', 'Size', 'Used MEM(MBytes)') )
        f.write("\n")
        tensors = [obj for obj in objects if torch.is_tensor(obj)]
        cuda_tensors = [t for t in tensors if t.is_cuda]
        host_tensors = [t for t in tensors if not t.is_cuda]
        _mem_report(cuda_tensors, 'GPU')
        _mem_report(host_tensors, 'CPU')
        f.write('='*LEN)
        f.write("\n")