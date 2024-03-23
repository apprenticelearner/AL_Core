# Find the area of the shaded region, (1/2)*(12km*20km), and then subtract the area of the unshaded region (1/2)*(6km*10km), from the total
def calc1():
    return (1/2)*(12*20) - (1/2)*(6*10)

# Find the area of the large triangle by multiplying the base 20km by height 12km and 1/2. Then find the area of the small triangle by multiplying base 10km by height 6km and 1/2. Subtract the area of the small triangle from the area of the big triangle.
def calc2():
    large_triangle = 20 * 12 * 1/2
    small_triangle = 10 * 6 * 1/2
    return large_triangle - small_triangle

# The area of the larger is half of 20 times 12. The area of the smaller triangle is half of 6 times 10. Subtract the area of the smaller triangle from the area of the larger triangle.
def calc3():
    return (20 * 12) / 2 - (6 * 10) / 2

# Multiply 12 and 20 and divide by two.  Then multiply 10 and 6 and divide that product by two.  Then subtract the smaller triangle area from the larger triangle area. 
def calc4():
    return (12*20/2) - (10*6/2)

# The area for the larger triangle is (one half times 20 km times 12 km which equals 120km squared) the smaller triangle's area is (one half times 6 km times 10 km which equals 30 km squared.) Then you subtract the small white area to leave the shade region which is 120 km squared minus 30 km squared.
def calc5():
    print(120-30)

# Find the area of the larger triangle by multiplying the base (20) by the width (12) and dividing that in half. Then do the same for the smaller triangle by multiplying the base (10) by the width (6) and dividing in half. Here, the rectangles have areas of 120 and 30. To find the area of the shaded region, subtract the area 30 from the area 120.
def calc6():
    area1 = (20 * 12) / 2
    area2 = (10 * 6) / 2
    print(area1 - area2)