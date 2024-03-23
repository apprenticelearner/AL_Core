# Calculate the ratio of the sector by dividing the 135 degree arc of the sector by 360, the total degree of a circle. Then, multiply this product by the square of the radius 6 to calculate the value multiplied by pi for the area of the sector 
def calc1():
    ratio = (135/360)
    area = ratio * (6**2)
    return area * 3.14  # 3.14 is the value of pi       

# Using the formula and plugging in our info we get 135 over 360 times pi 6 squared which equals  13.5 pi cm squared
def calc2():
    return 135/360*3.14*6**2