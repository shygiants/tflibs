""" Device setter """


def device_setter(device):
    cpu = '/cpu:0'

    def device_fn(op):
        cpu_ops = ['Trainable_Variables', 'RandomShuffle', 'Summary', 'Const']
        return cpu if any(map(lambda k: k in op.name or k in op.type, cpu_ops)) else device

    return device_fn if device else cpu
