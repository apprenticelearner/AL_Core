# Add 8 and 4. Since it is more than 10, take just the ones digit.
def calc1():
    return (8+4)%10

# Add the digits in the tens place, 8 and 4. Since the sum exceeds 10, write down the ones of the sum (2), then carry the tens (1) to the hundreds place.
def calc2():
    x = 8
    y = 4
    z = x + y
    print(z)    # 12

# The sum of 8 and 4 without the 1. 
def calc3():
    print(8 + 4 - 1)    

# add 8 to 4 which gives you 12 and you put down the ones column, which is a two and carry the one over to the next column to the left.
def calc4():
    print(8+4)  

# Add 8 plus 4. Bring down the answer's ones place digit, next to the 9. Carry the tens place digit up above the 3 to be added in the hundreds place.
def calc5():
    print(8 + 4)
    print(" Bring down the answer's ones place digit, next to the 9. Carry the tens place digit up above the 3 to be added in the hundreds place.")

# Add the numeral 8 plus the numeral 4, and then use the last digit of the resulting sum (12) to find the missing value (2).
def calc6():
    return 8 + 4 - 12   # 2
