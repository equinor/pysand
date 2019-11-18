class FunctionInputFail(Exception):
    """
    Raise when input data is configured in a way which makes calculating a result impossible or meaningless
    (I.e. constant value = None, Missing timeseries etc)
    """
    pass
