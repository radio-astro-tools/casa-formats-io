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
        return self


class RecordDesc(BaseCasaObject):

    @classmethod
    @with_nbytes_prefix
    def read(cls, f):

        self = cls()

        check_type_and_version(f, 'RecordDesc', 2)

        self.nrec = read_int32(f)

        self.names = []
        self.types = []
        self.values = []

        for i in range(self.nrec):
            self.names.append(read_string(f))
            self.types.append(TYPES[read_int32(f)])
            # Here we don't actually load in the data for may of the types - hence
            # why we don't do anything with the values we read in.
            if self.types[-1] in ('bool', 'int', 'uint', 'float', 'double',
                                  'complex', 'dcomplex', 'string'):
                self.values.append(read_string(f))  # noqa
            elif self.types[-1] == 'table':
                self.values.append(f.read(8))
            elif self.types[-1].startswith('array'):
                self.values.append(read_iposition(f))
                self.values.append(f.read(4))
            elif self.types[-1] == 'record':
                self.values.append(RecordDesc.read(f))
                self.values.append(read_int32(f))
            else:
                raise NotImplementedError("Support for type {0} in RecordDesc "
                                          "not implemented".format(self.types[-1]))

        return self
