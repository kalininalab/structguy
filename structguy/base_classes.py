from structguy import dicts

from structman.base_utils.base_utils import Slotted_obj

possible_na_values = set(['-', 'None', 'inf'])
class Feature(Slotted_obj):
    __slots__ = ['name', 'f_type', 'group', 'default', 'mutation_specific', 'category_map', 'category_counter', 'category_backmap']
    def __init__(self, name = None, f_type = None,group=None,default_value=None,mutation_specific=False):
        self.name = name
        self.f_type = f_type
        self.group = group
        self.default = default_value
        self.mutation_specific=mutation_specific

        if f_type == 'categorical':
            self.category_map = {}
            self.category_counter = 0
            self.category_backmap = {}


    def value_from_string(self, string):
        value = None
        if string in possible_na_values:
            return None

        if self.name == 'Blosum62':
            try:
                value = float(string)
            except ValueError:
                try:
                    value = dicts.BLOSUM62[(string[0],string[-1])]
                except KeyError:
                    value = dicts.BLOSUM62[(string[-1],string[0])]
            return value

        try:
            if self.f_type == 'categorical':
                value = string
            elif self.f_type == 'real':
                value = float(string)
            elif self.f_type == 'integer' or self.f_type == 'binary':
                try:
                    value = int(string)
                except ValueError:
                    value = float(string)
                    self.f_type = 'real'
            elif self.f_type == 'unknown':
                try:
                    value = int(string)
                    if value != self.default:
                        self.f_type = 'integer'
                except:
                    try:
                        value = float(string)
                        if value != self.default:
                            self.f_type = 'real'
                    except:
                        self.f_type = 'categorical'
                        self.category_map = {}
                        self.category_counter = 0
                        self.category_backmap = {}

                        value = string
            else:
                print(f'Error in value_from_string: {self.f_type=} {self.name=} {string=}')
                value = None
        except:
            print(f'Error in value_from_string: {self.f_type=} {self.name=} {string=}')
            value = None
        return value

    def string_convert(self,value):
        if not self.f_type == 'categorical':
            return str(value)
        elif value is None:
            return 'None'
        else:
            return str(self.category_backmap[value])
            #return str(value)
