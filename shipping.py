import iso6346

class ShippingContainer:
    
    next_serial = 1337
    
    @staticmethod
    def _make_bic_code(owner_code,serial):
        return iso6346.create(owner_code=owner_code,serial=str(serial).zfill(6))
    
    #def _get_next_serial(self):
    #@staticmethod
    #def _get_next_serial():
    #    result = ShippingContainer.next_serial
    #    ShippingContainer.next_serial += 1
    #    return result

    @classmethod
    def _get_next_serial(cls):
        result = cls.next_serial
        cls.next_serial += 1
        return result
    
    @classmethod
    def create_empty(cls,owner_code):
        return cls(owner_code,contents=None)

    @classmethod
    def create_with_items(cls,owner_code,items):
        return cls(owner_code,contents=list(items))
    
    def __init__(self,owner_code,contents):
        self.owner_code = owner_code
        self.contents = contents 
        self.bic = ShippingContainer._make_bic_code(owner_code=owner_code,
                                                    serial=ShippingContainer._get_next_serial())
        #self.serial = self._get_next_serial()
        #self.serial = ShippingContainer.next_serial
        #ShippingContainer.next_serial += 1
        
        
        
        
c1 = ShippingContainer("YML","books")
print(c1.owner_code)
print(c1.contents)
#print(c1.serial)
print(c1.bic)

c2 = ShippingContainer("MAE","tools")
print(c2.owner_code)
print(c2.contents)
#print(c2.serial)
print(c2.bic)
#print(ShippingContainer.next_serial)
#print(c2.next_serial)
#print(c1.next_serial)
c3 = ShippingContainer.create_empty("ABC")
print(c3.owner_code)
print(c3.contents)
#print(c3.serial)
print(c3.bic)

c4 = ShippingContainer.create_with_items("EFG",['food','tools'])
print(c4.owner_code)
print(c4.contents)
#print(c4.serial)
print(c4.bic)