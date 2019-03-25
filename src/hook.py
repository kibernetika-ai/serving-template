
def init_hook(**kwargs):
    print('init hook')
    print(kwargs)


def preprocess(inputs, ctx):
    print('preprocess')
    return inputs


def postprocess(outputs, ctx):
    print('postprocess')
    return outputs
