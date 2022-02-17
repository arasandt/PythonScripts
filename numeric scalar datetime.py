# =============================================================================
# import decimal
# from decimal import Decimal
# decimal.getcontext().prec = 6
# d = Decimal('1.234567')
# print(d)
# d += Decimal('1')
# print(d)
# =============================================================================


from fractions import Fraction
a = Fraction(2,3)
print(a)

a = Fraction(0.5)
print(a)

a = 2j
print(type(a))


import datetime

d = datetime.date.today()
print("{date:%A} {date.day} {date:%B} {date.year}".format(date=d))

