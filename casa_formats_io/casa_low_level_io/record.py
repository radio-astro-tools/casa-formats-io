from .core import (check_type_and_version, BaseCasaObject, with_nbytes_prefix,
                   read_string, read_int32, read_iposition, TYPES)


class Record(BaseCasaObject):

    @classmethod
    @with_nbytes_prefix
    def read(cls, f):
        self = cls()
        check_type_and_version(f, 'Record', 1)
        self.desc = RecordDesc.read(f)
        read_int32(f)  # Not sure what the following value is


class RecordDesc(BaseCasaObject):

    @classmethod
    @with_nbytes_prefix
    def read(cls, f):

        self = cls()

        check_type_and_version(f, 'RecordDesc', 2)

        nrec = read_int32(f)

        self.names = []
        self.types = []

        for i in range(nrec):
            self.names.append(read_string(f))
            self.types.append(TYPES[read_int32(f)])
            # Here we don't actually load in the data for may of the types - hence
            # why we don't do anything with the values we read in.
            if self.types[-1] in ('bool', 'int', 'uint', 'float', 'double',
                                  'complex', 'dcomplex', 'string'):
                comment = read_string(f)  # noqa
            elif self.types[-1] == 'table':
                f.read(8)
            elif self.types[-1].startswith('array'):
                read_iposition(f)
                f.read(4)
            elif self.types[-1] == 'record':
                RecordDesc.read(f)
                read_int32(f)
            else:
                raise NotImplementedError("Support for type {0} in RecordDesc "
                                          "not implemented".format(self.types[-1]))

        return self
