# Divide 12 by 3 and multiply that value by 2
def calc1():
    return 2 * (12 / 3)

# The larger triangle has a ratio of 3 over 12. The smaller triangle has a ratio of 2 over x. If you set the ratios equal to each other (because they are similar triangles), you can use cross-multiplication to solve for x. Multiply 2 and 12. Divide that value by 3.
def calc2():
    return (2*12)/3

# Divide 3 by 2 and use the quotient to divide 12 by. 
def calc3():
    # Divide 3 by 2
    quotient = 3 / 2
    # Use the quotient to divide 12 by
    quotient2 = 12 / quotient
    # Print the result
    print(quotient2)

# 12 divided by 3 is 4 so the green triangle's longer side is 4 times larger than its smaller one. Now you take the 2 on the smaller side of the pink triangle and multiply it by 4 to get 8 yd. 
def calc4():
    print("12 divided by 3 is 4 so the green triangle's longer side is 4 times larger than its smaller one. Now you take the 2 on the smaller side of the pink triangle and multiply it by 4 to get 8 yd.")
    print("The answer is 8 yd.")
    print("")   

# Use the length of the two similar sides to make a ratio. In this case, use 2 and 3 yards to make the ratio, 2/3. Make the ratio with the other two similar sides, in this case x and 12, to make the ratio x/12. Cross-multiply the two ratios and solve for x. 
def calc5():
    print("2/3 = x/12")
    print("2x = 36")
    print("x = 18")
    print("The length of the side is 18 yards.")    

# To find the length of side x, divide 3 by 2 to get 1.5. Take 1.5 and multiply it with 12
def calc6():
    x = (3/2)*12
    print(x)
