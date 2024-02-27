import numpy as np
import sys
import time
import mmap
import ctypes
import posix_ipc

from string import ascii_letters, digits

valid_chars = frozenset("/-_. %s%s" % (ascii_letters, digits))

typecode_to_type = {
    'c': ctypes.c_char, 'u': ctypes.c_wchar,
    'b': ctypes.c_byte, 'B': ctypes.c_ubyte,
    'h': ctypes.c_short, 'H': ctypes.c_ushort,
    'i': ctypes.c_int, 'I': ctypes.c_uint,
    'l': ctypes.c_long, 'L': ctypes.c_ulong,
    'f': ctypes.c_float, 'd': ctypes.c_double
}

def address_of_buffer(buf):  # (python 3)
    return ctypes.addressof(ctypes.c_char.from_buffer(buf))

class ShmemBufferWrapper:

    def __init__(self, tag, size, create=True):
        # default vals so __del__ doesn't fail if __init__ fails to complete
        self._mem = None
        self._map = None
        self._owner = create
        self.size = size

        assert 0 <= size < sys.maxsize  # sys.maxint  (python 3)
        flag = (0, posix_ipc.O_CREX)[create]
        mem_size = (0, self.size)[create]

        self._mem = posix_ipc.SharedMemory(tag, flags=flag, size=mem_size)
        self._map = mmap.mmap(self._mem.fd, self._mem.size)
        self._mem.close_fd()

    def get_address(self):
        # assert self._map.size() == self.size  # (changed for python 3)
        assert self._map.size() >= self.size # strictly equal might not meet in MAC
        addr = address_of_buffer(self._map)
        return addr

    def __del__(self):
        if self._map is not None:
            self._map.close()
        if self._mem is not None and self._owner:
            self._mem.unlink()

def ShmemRawArray(typecode_or_type, size_or_initializer, tag, create=True):
    assert frozenset(tag).issubset(valid_chars)
    if tag[0] != "/":
        tag = "/%s" % (tag,)

    type_ = typecode_to_type.get(typecode_or_type, typecode_or_type)
    if isinstance(size_or_initializer, int):
        type_ = type_ * size_or_initializer
    else:
        type_ = type_ * len(size_or_initializer)

    buffer = ShmemBufferWrapper(tag, ctypes.sizeof(type_), create=create)
    obj = type_.from_address(buffer.get_address())
    obj._buffer = buffer

    if not isinstance(size_or_initializer, int):
        obj.__init__(*size_or_initializer)

    return obj

def NpShmemArray(shape, dtype, tag, create=True):
    size = int(np.prod(shape))
    nbytes = size * np.dtype(dtype).itemsize
    shmem = ShmemRawArray(ctypes.c_char, nbytes, tag, create)
    return np.frombuffer(shmem, dtype=dtype, count=size).reshape(shape)

def get_random_tag():
    return str(time.time()).replace(".", "")[-9:]