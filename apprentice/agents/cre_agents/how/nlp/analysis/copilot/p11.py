# Divide 144 by the sum of 1 and .2 to get the number of pears Type B produced
def calc1():
    return 144/(1+.2)

# The total percentage value is 100 plus 20. Divide 144, the number of Type A pears, by 1.2.
def calc2():
    return 100 + 20 / 144 / 1.2 

# a equals b plus 20 percent so 120 percent. b equals 100 percent. 10 divided by 120 equals 83.33 percent. So b produces 83.33 percent (.8333) of a (144) which equals 120
def calc3():
    a = 144
    b = a * .8333
    return b