def _peek(f, length):
    # Internal function used for debugging - show the next N bytes (and the
    # previous 8)
    pos = f.tell()
    f.seek(pos - 8)
    print(repr(f.read(8)), '  ', repr(f.read(length)))
    f.seek(pos)


def _peek_no_prefix(f, length):
    # Internal function used for debugging - show the next N bytes
    pos = f.tell()
    print(repr(f.read(length)))
    f.seek(pos)
