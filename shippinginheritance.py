
import iso6346

class ShippingContainer:
    
    next_serial = 1337
    
    HEIGHT_FT = 8.5
    WIDTH_FT = 8.0
    
    @staticmethod
    def _make_bic_code(owner_code,serial):
        return iso6346.create(owner_code=owner_code,serial=str(serial).zfill(6))
    
    @classmethod
    def _get_next_serial(cls):
        result = cls.next_serial
        cls.next_serial += 1
        return result
    
    @classmethod
    def create_empty(cls,owner_code,*args,**kwargs):
        return cls(owner_code,contents=None,*args,**kwargs)

    @classmethod
    def create_with_items(cls,owner_code,items,*args,**kwargs):
        return cls(owner_code,contents=list(items),*args,**kwargs)
    
    def __init__(self,owner_code,length_ft,contents):
        self.owner_code = owner_code
        self.contents = contents 
        self.length_ft = length_ft 
        #self.bic = ShippingContainer._make_bic_code(owner_code=owner_code,
        self.bic = self._make_bic_code(owner_code=owner_code,
                                                    serial=ShippingContainer._get_next_serial())
    def _calc_volume(self):
        return ShippingContainer.HEIGHT_FT * ShippingContainer.WIDTH_FT * self.length_ft

    @property
    def volume_ft3(self):
        return self._calc_volume()

      
class RefShippingContainer(ShippingContainer):
    
    MAX_CELSIUS = 4.0
    
    FRIDGE_VOLUME_FT3 = 100.0
    
    @staticmethod
    def _make_bic_code(owner_code,serial):
        return iso6346.create(owner_code=owner_code,serial=str(serial).zfill(6),category="R")
    
    def __init__(self,owner_code,length_ft,contents,celsius):
        super().__init__(owner_code,length_ft,contents)
        #if celsius > RefShippingContainer.MAX_CELSIUS:
        #    raise ValueError("Temp is too hot!")
        self.celsius = celsius
        #self._celsius = celsius
    
    @property #instead of getter and setter
    def celsius(self):
        return self._celsius
        
    @celsius.setter
    def celsius(self,value):
        self._set_celsius(value)

    def _set_celsius(self,value):
        if value > RefShippingContainer.MAX_CELSIUS:
            raise ValueError('Temp is too hot!')
        self._celsius = value

    @staticmethod
    def _c_to_f(celsius):
        return celsius * 9/5 + 32
    
    @staticmethod
    def _f_to_c(fahrenheit):
        return (fahrenheit - 32) * 5/9 
    
    @property #instead of getter and setter
    def fahrenheit(self):
        return RefShippingContainer._c_to_f(self._celsius)
        
    @fahrenheit.setter
    def fahrenheit(self,value):
        self.celsius = RefShippingContainer._f_to_c(value)

    #@property
    def _calc_volume(self):
        return super()._calc_volume() - RefShippingContainer.FRIDGE_VOLUME_FT3
        #return ShippingContainer.HEIGHT_FT * ShippingContainer.WIDTH_FT * self.length_ft - RefShippingContainer.FRIDGE_VOLUME_FT3


class HeatedRefShippingContainer(RefShippingContainer):
     
    MIN_CELSIUS = -20.0
     
# =============================================================================
#     @RefShippingContainer.celsius.setter
#     def celsius(self,value):
#         #if not (HeatedRefShippingContainer.MIN_CELSIUS <= value <= RefShippingContainer.MAX_CELSIUS):
#         #    raise ValueError('Temp is out of range!')
#         if value < HeatedRefShippingContainer.MIN_CELSIUS:
#             raise ValueError('Temp is too cold!')
#         #self._celsius = value
#         #super().celsius = value -- does not work
#         RefShippingContainer.celsius.fset(self,value)
# 
# =============================================================================
    def _set_celsius(self,value):
        if value < HeatedRefShippingContainer.MIN_CELSIUS:
            raise ValueError('Temp is too cold!')
        super()._set_celsius(value)


c1 = HeatedRefShippingContainer.create_empty("YML",length_ft=20,celsius=-15)
print(c1.celsius)
c1.celsius = -10
print(c1.celsius)
c1.fahrenheit = -14
print(c1.fahrenheit)


# =============================================================================
# c1 = ShippingContainer.create_empty("YML",length_ft=20)
# print(c1.volume_ft3)
# 
# c2 = RefShippingContainer.create_empty("YML",length_ft=10,celsius=-20.0)
# print(c2.volume_ft3)
# 
# =============================================================================
        
# =============================================================================
# c4 = RefShippingContainer.create_with_items("EFG",['food','tools'],celsius=-20.0)
# print(c4.owner_code)
# print(c4.contents)
# print(c4.bic)
# print(c4.celsius)
# print(c4.fahrenheit)
# #c4.celsius = -19.0
# #print(c4.celsius)
# c4.fahrenheit = -10.0
# print(c4.fahrenheit)
# print(c4.celsius)
# 
# =============================================================================
# 
# =============================================================================
# c5 = RefShippingContainer("MAE","1books")
# print(c5.owner_code)
# print(c5.contents)
# print(c5.bic)
#  
# =============================================================================

# =============================================================================
# c1 = ShippingContainer("YML","books")
# print(c1.owner_code)
# print(c1.contents)
# print(c1.bic)
# 
# c2 = ShippingContainer("MAE","tools")
# print(c2.owner_code)
# print(c2.contents)
# print(c2.bic)
# 
# c3 = ShippingContainer.create_empty("ABC")
# print(c3.owner_code)
# print(c3.contents)
# print(c3.bic)
# 
# c4 = ShippingContainer.create_with_items("EFG",['food','tools'])
# print(c4.owner_code)
# print(c4.contents)
# print(c4.bic)
# 
# =============================================================================




















