# Subtract 8 by 4 and Subtract 7 by 5. Divide the first sum by the second sum.
def calc1():
    return (8-4)/(7-5)  # 1.0

# The slope is the quotient of the difference between the y-values 8 and 4, and the difference between the x-values 7 and 5.
def calc2():
    print((8-4)/(7-5))  # 1.0

# subtract 7 and 5, then divide the product by 8 minus 4
def calc3():
    return (7-5)/(8-4)  

# You take y2 which is 8 and subtract y1 which is 4 and get 4. Put that aside for a minute and take x2 (7) and subtract x1 (5) which is 2. Take the four from the differences in the y values and put that over the two from the difference of x values and you get 4 over 2 which is 2.
def calc4():
    y2 = 8
    y1 = 4
    x2 = 7
    x1 = 5
    return (y2 - y1) / (x2 - x1)

# The change in our x points if found by subtracting 5 from 7. The change in our y points is found by subtracting 4 from 8. Divide the change in y by the change x to find the slope.
def calc5():
    x1 = 5
    x2 = 7
    y1 = 4
    y2 = 8
    slope = (y2-y1)/(x2-x1)
    print(slope)    

# Subtract the first y-coordinate 4 from the second y-coordinate 8, then subtract the first x-coordinate 5 from the second x-coordinate 7, and then divide the difference of the y-coordinates (4) by the difference of the x-coordinates (2).
def calc6():
    y = 8 - 4
    x = 7 - 5
    return y / x    

# Take 4 and subtract it from 8 to get 4. Take 5 and subtract it from 7. Take 4 divided by 2 to get 2
def calc7():
    a = 8 - 4
    b = 7 - 5
    c = 4 / 2
    return a, b, c  # return a tuple of values 
