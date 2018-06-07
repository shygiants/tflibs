""" Summary """


def strip_illegal_summary_name(name):
    """
    Strips illegal summary name
    :param str name:
    :return: Stripped name
    """
    return name.rstrip(':0')
