stats = {}
def codex_eval(num, problem, student, goal, _fn_correct=None): 
    
    data = stats.get(num, {**{
        'problem': problem,
        'student': student,
        'fn_correct' : 0,
        'resp_correct' : 0,
        'total' : 0
    }})
    
    def wrapper(fn):
        fn_correct = _fn_correct
        val = fn()
        corr_resp = (val == goal)

        print(num, val, corr_resp)
        if(fn_correct is None):
            fn_correct = corr_resp

        if(fn_correct):
            data['fn_correct'] += 1 

        if(corr_resp):
            data['resp_correct'] += 1 

        data['total'] += 1
        stats[num] = data
    return wrapper


def show_stats(num):
    has_correct = stats[num]['fn_correct'] > 0
    n_incorrect = stats[num]['total'] - stats[num]['fn_correct']
    n_bad_fn = stats[num]['resp_correct'] - stats[num]['fn_correct']
    problem = stats[num]['problem']
    student = stats[num]['student']
    print(f"{num} (p: {problem}, s: {student}) : ({int(has_correct)}/{n_incorrect+has_correct}/{n_bad_fn+has_correct}))")

# ---------------------------------------------------------
# 1: ((3-8) / Double(2)) + 1) == -0.25
# "Divide 3 minus 8 and twice 2, then increment."

# @codex_eval(1, -0.25)
# def foo():
#     ''''Divide 3 minus 8 and twice 2, then increment.'''
#     return 3 - 8 / 2 * 2 + 1

# @codex_eval(1, -0.25)
# def foo():
#     ''''Divide 3 minus 8 and twice 2, then increment.'''
#     return (3 - 8) / 2 * 2 + 1

# @codex_eval(1, -0.25)
# def foo():
#     ''''Divide 3 minus 8 and twice 2, then increment.'''
#     return (3 - 8) / (2 * 2) + 1

# @codex_eval(1, -0.25, False)
# def foo():
#     ''''Divide 3 minus 8 and twice 2, then increment.'''
#     return ((3 - 8) / 2) * 2 + 1

# show_stats(1)


# ---------------------------------------------------------
# 1, p: 2, s: 0
# Multiply 3x and 2x, the first two terms of each binomial and take the constant

@codex_eval(1, 2, 0, 6)
def calc():
    a = 3
    b = 2
    c = 1
    d = 1
    e = 1
    f = 1
    g = 1
    h = 1
    i = 1
    j = 1
    k = 1
    l = 1
    m = 1
    n = 1
    o = 1
    p = 1
    q = 1
    r = 1
    s = 1
    t = 1
    u = 1
    v = 1
    w = 1
    x = 1
    y = 1
    z = 1
    return (a*b*c*d*e*f*g*h*i*j*k*l*m*n*o*p*q*r*s*t*u*v*w*x*y*z)

@codex_eval(1, 2, 0, 6)
def calc():
    return 3 * 2 * 4 * 3 * 5 * 4 * 6 * 5 * 7 * 6 * 8 * 7 * 9 * 8 * 10 * 9 * 11 * 10 * 12 * 11 * 13 * 12 * 14 * 13 * 15 * 14 * 16 * 15 * 17 * 16 * 18 * 17 * 19 * 18 * 20 * 19 * 21 * 20 * 22 * 21 * 23 * 22 * 24 * 23 * 25 * 24 * 26 * 25 * 27 * 26 * 28 * 27 * 29 * 28 * 30 * 29 * 31 * 30 * 32 * 31 * 33 * 32 * 34 * 33 * 35 * 34 * 36 * 35 * 37 * 36 * 38 * 37 * 39 * 38 * 40 * 39 * 41 * 40 * 42 * 41 * 43 * 42 * 44 * 43 * 45 * 44 * 46 * 45 * 47 * 46 * 48 * 47 * 49 * 48 * 50 * 49 * 51 * 50 * 52 * 51 * 53 * 52 * 54 * 53 * 55 * 54 * 56 * 55 * 57 * 56 * 58 * 57 * 59 * 58 * 60 * 59 * 61 * 60 * 62 * 61 * 63 * 62 * 64 * 63 * 65 * 64 * 66 * 65 * 67 * 66 * 68 * 67 * 69 * 68 * 70 * 69 * 71 * 70 * 72 * 71 * 73 * 72 * 74 * 73 * 75 * 74 * 76 * 75 * 77 * 76 * 78 * 77 * 79 * 78 * 80 * 79 * 81 * 80 * 82 * 81 * 83 * 82 * 84 * 83 * 85 * 84

@codex_eval(1, 2, 0, 6)
def calc():
    x = 1
    y = 2
    z = 3
    return (x * y) + (x * z) + (y * z)

@codex_eval(1, 2, 0, 6)
def calc():
    return (3*2) + (2*1)

@codex_eval(1, 2, 0, 6)
def calc():
    return (3*2*1)*(2*1*1)

@codex_eval(1, 2, 0, 6)
def calc():
    return 3 * 2 * 6 + 3 * 2 * 4

@codex_eval(1, 2, 0, 6)
def calc():
    return 3*2*2*1

@codex_eval(1, 2, 0, 6)
def calc():
    pass

@codex_eval(1, 2, 0, 6)
def calc():
    return 3*2*2

show_stats(1)

# ---------------------------------------------------------
# 2, p: 4, s: 0
# Multiply 4 and 5 to get c

@codex_eval(2, 4, 0, 20)
def calc():
    a = 4
    b = 5
    c = a * b
    return c

@codex_eval(2, 4, 0, 20)
def calc():
    c = 4 * 5
    return c

@codex_eval(2, 4, 0, 20)
def calc():
    return 4 * 5

show_stats(2)



# ---------------------------------------------------------
# 3, p: 5, s: 0
# Multiply 4 and 3 to find the denominator of the new fractions

@codex_eval(3, 5, 0, 12)
def calc():
    return

@codex_eval(3, 5, 0, 12)
def calc():
    return 4 * 3

show_stats(3)


# ---------------------------------------------------------
# 4, p: 6, s: 0
# Multiply the 3 of the first fraction by the 3 needed to turn the "4" denominator into 12

@codex_eval(4, 6, 0, 9)
def calc():
    return (3/4) * (12/3)

@codex_eval(4, 6, 0, 9)
def calc():
    return 3 * (12/4)

@codex_eval(4, 6, 0, 9)
def calc():
    return 3 * 12 / 4

@codex_eval(4, 6, 0, 9)
def calc():
    return 3 * 3

@codex_eval(4, 6, 0, 9)
def calc():
    pass
    # Your code goes here
    # Don't forget to return the result

@codex_eval(4, 6, 0, 9)
def calc():
    return (3*4)/12

@codex_eval(4, 6, 0, 9)
def calc():
    return 1/4 * 3 * 3

show_stats(4)


# ---------------------------------------------------------
# 5, p: 7, s: 0
# Add 9 and 8 to get the new numerator

@codex_eval(5, 7, 0, 17)
def calc():
    return 9 + 8

@codex_eval(5, 7, 0, 17, False)
def calc():
    # Write your code here
    return 17

show_stats(5)




# ---------------------------------------------------------
# 6, p: 11, s: 0
# Divide 144 by the sum of 1 and .2 to get the number of pears Type B produced

@codex_eval(6, 11, 0, 120)
def calc():
    return 144 / (1 + .2)

@codex_eval(6, 11, 0, 120)
def calc():
    return 144 / (1 + 0.2)

show_stats(6)

# ---------------------------------------------------------
# 7, p: 12, s: 0
# Divide 10 by 2 first and then add 3.
@codex_eval(7, 12, 0, 8)
def calc():
    return 10 / 2 + 3

@codex_eval(7, 12, 0, 8)
def calc():
    return (10 / 2) + 3


show_stats(7)


# ---------------------------------------------------------
# 8, p: 13, s: 0
# Add 8 and 4. Since it is more than 10, take just the ones digit.
@codex_eval(8, 13, 0, 2)
def calc():
    return (8 + 4) % 10

@codex_eval(8, 13, 0, 2)
def calc():
    return 8 + 4 % 10

show_stats(8)


# ---------------------------------------------------------
# 9, p: 14, s: 0
# Subtract 8 by 4 and Subtract 7 by 5. Divide the first sum by the second sum.

@codex_eval(9, 14, 0, 2)
def calc():
    a = 8 - 4
    b = 7 - 5
    return a/b

@codex_eval(9, 14, 0, 2)
def calc():
    return (4-8)/(5-7)

@codex_eval(9, 14, 0, 2)
def calc():
    return (8-4)/(7-5)

@codex_eval(9, 14, 0, 2)
def calc():
    return ((8-4)/(7-5))

@codex_eval(9, 14, 0, 2)
def calc():
    return (4-8)/(7-5)


show_stats(9)


# ---------------------------------------------------------
# 10, p: 1, s: 1
# 3 divided by 12 is .25.  .25 multiplied by 100 is 25%

@codex_eval(10, 1, 1, 25)
def calc():
    return 3/12*100

@codex_eval(10, 1, 1, 25, False)
def calc():
    return 25

@codex_eval(10, 1, 1, 25)
def calc():
    return (3 / 12) * 100

@codex_eval(10, 1, 1, 25)
def calc():
    return 3/12


show_stats(10)



# ---------------------------------------------------------
# 11, p: 7, s: 1
# Add the values of the numerators in the fractions 9/12 and 8/12.

@codex_eval(11, 7, 1, 17)
def calc():
    pass

@codex_eval(11, 7, 1, 17)
def calc():
    return 9/12 + 8/12

@codex_eval(11, 7, 1, 17)
def calc():
    return 17/12

@codex_eval(11, 7, 1, 17)
def calc():
    return 9+8

show_stats(11)


# ---------------------------------------------------------
# 12, p: 10, s: 1
# Divide 12 by 3 and multiply that value by 2


@codex_eval(12, 10, 1, 8)
def calc():
    return 12 / 3 * 2

@codex_eval(12, 10, 1, 8)
def calc():
    return (12 / 3) * 2

@codex_eval(12, 10, 1, 8)
def calc():
    return 2 * (12 / 3)

show_stats(12)  



# ---------------------------------------------------------
# 13, p: 12, s: 1
# Divide 10/2 and then add 3

@codex_eval(13, 12, 1, 8)
def calc():
    return 10/2 + 3

@codex_eval(13, 12, 1, 8)
def calc():
    return (10/2)+3

@codex_eval(13, 12, 1, 8)
def calc():
    return ((10/2) + 3)

show_stats(13)



# ---------------------------------------------------------
# 14, p: 1, s: 2
# Form a ratio with 3 as the numerator, 12 as the denominator, and convert the decimal into a percent by multiplying 100

@codex_eval(14, 1, 2, 25)
def calc():
    ratio = 3/12
    percent = ratio * 100
    return percent

@codex_eval(14, 1, 2, 25)
def calc():
    return (3 / 12) * 100

@codex_eval(14, 1, 2, 25)
def calc():
    return 3/12 * 100

@codex_eval(14, 1, 2, 25)
def calc():
    return 3.0/12.0*100

show_stats(14)


# ---------------------------------------------------------
# 15, p: 2, s: 2
# a is the product of integer 3 and 2, which are both in front of variable x.

@codex_eval(15, 2, 2, 6)
def calc():
    x = 23 # I add this otherwise this cannot be run.
    a = 3 * 2 * x
    return a

@codex_eval(15, 2, 2, 6)
def calc():
    x = 1
    a = 3 * 2 * x
    return a

@codex_eval(15, 2, 2, 6)
def calc():
    x = 1
    a = 3 * 2 * x
    print(a)
    return a

@codex_eval(15, 2, 2, 6)
def calc():
    a = 3 * 2
    x = 1
    return a * x

show_stats(15)



# ---------------------------------------------------------
# 16, p: 4, s: 2
# c is calculated by multiplying 4 and 5, the constant terms

@codex_eval(16, 4, 2, 20)
def calc():
    a = 4
    b = 5
    c = a * b
    return c

@codex_eval(16, 4, 2, 20)
def calc():
    a = 4
    b = 5
    c = a * b
    print(c)

@codex_eval(16, 4, 2, 20)
def calc():
    a = 1
    b = 2
    c = 4 * 5
    return a + b + c

@codex_eval(16, 4, 2, 20)
def calc():
    c = 4 * 5
    return c

show_stats(16)


# ---------------------------------------------------------
# 17, p: 6, s: 2
# Since the denominator 4 is converted to denominator 12 by multiplying 3, also multiply the numerator 3 by 3

@codex_eval(17, 6, 2, 9)
def calc():
    return 3 * 3 / 4

@codex_eval(17, 6, 2, 9)
def calc():
    a = 3*3
    b = 4*3
    c = a/b
    return c

@codex_eval(17, 6, 2, 9)
def calc():
    global a, b, c, d, e, f
    a = 1
    b = 3
    c = 3
    d = 12
    e = 2
    f = 3

@codex_eval(17, 6, 2, 9)
def calc():
    return 3 * 3, 12

@codex_eval(17, 6, 2, 9)
def calc():
    return 3.0 * 3 / 12


show_stats(17)



# ---------------------------------------------------------
# 18, p: 7, s: 2
# Since the two fractions share the same denominator 12, simply add the numerators 9 and 8 to calculate the sum of the fraction

@codex_eval(18, 7, 2, 17)
def calc():
    a = 9
    b = 12
    c = 8
    d = 12
    e = a + c
    f = b
    print(e, '/', f)

@codex_eval(18, 7, 2, 17)
def calc():
    # The numerator of the fraction is 9
    num = 9
    # The denominator of the fraction is 12
    denom = 12
    # The numerator of the second fraction is 8
    num2 = 8
    # The denominator of the second fraction is 12
    denom2 = 12
    # Add the numerators of the two fractions and store the result in the variable sum
    sum = num + num2
    # Print the sum of the two fractions
    print(sum)

@codex_eval(18, 7, 2, 17)
def calc():
    return 9 + 8

@codex_eval(18, 7, 2, 17)
def calc():
    print(9+8)
    return 0

show_stats(18)

# ---------------------------------------------------------
# 19, p: 8, s: 2
# Find the area of the large triangle by multiplying the base 20km by height 12km and 1/2. Then find the area of the small triangle by multiplying base 10km by height 6km and 1/2. Subtract the area of the small triangle from the area of the big triangle.

@codex_eval(19, 8, 2, 90)
def calc():
    area1 = 20 * 12 * 0.5
    area2 = 10 * 6 * 0.5
    area = area1 - area2

@codex_eval(19, 8, 2, 90)
def calc():
    large_triangle = 20 * 12 * 1/2
    small_triangle = 10 * 6 * 1/2
    return large_triangle - small_triangle

@codex_eval(19, 8, 2, 90)
def calc():
    large = 20 * 12 * 1/2
    small = 10 * 6 * 1/2
    return large - small

@codex_eval(19, 8, 2, 90)
def calc():
    return (20 * 12 * 0.5) - (10 * 6 * 0.5)

@codex_eval(19, 8, 2, 90)
def calc():
    large_triangle = ((20 * 12) * 0.5)
    small_triangle = ((10 * 6) * 0.5)
    return large_triangle - small_triangle

@codex_eval(19, 8, 2, 90)
def calc():
    bigTriangle = 20*12*0.5
    smallTriangle = 10*6*0.5
    return bigTriangle - smallTriangle

@codex_eval(19, 8, 2, 90)
def calc():
    return (20*12)/2 - (10*6)/2

@codex_eval(19, 8, 2, 90)
def calc():
    return ((20*12)/2) - ((10*6)/2)

@codex_eval(19, 8, 2, 90)
def calc():
    return 120 - 30


show_stats(19)




# ---------------------------------------------------------
# 20, p: 9, s: 2
# Calculate the ratio of the sector by dividing the 135 degree arc of the sector by 360, the total degree of a circle. Then, multiply this product by the square of the radius 6 to calculate the value multiplied by pi for the area of the sector 
@codex_eval(20, 9, 2, 13.5)
def calc():
    radius = 6
    angle = 135
    ratio = angle/360
    area = ratio*(radius**2)
    return area

@codex_eval(20, 9, 2, 13.5)
def calc():
    ratio = 135/360
    area = ratio * (6**2)
    return area

@codex_eval(20, 9, 2, 13.5)
def calc():
    ratio = 135/360
    area = ratio * 6**2
    return area

@codex_eval(20, 9, 2, 13.5)
def calc():
    ratio = 135 / 360
    return ratio * 6**2

@codex_eval(20, 9, 2, 13.5)
def calc():
    area = (135/360)*(6**2)
    return area

@codex_eval(20, 9, 2, 13.5)
def calc():
  radius = 6
  sector = 135
  ratio = sector/360
  area = radius**2 * ratio
  print("The area of the sector is", area)

@codex_eval(20, 9, 2, 13.5)
def calc():
    ratio = (135/360)
    pi = 3.14
    area = (6**2)*ratio

@codex_eval(20, 9, 2, 13.5)
def calc():
    return (135/360)*6**2

@codex_eval(20, 9, 2, 13.5)
def calc():
    return (135/360)*36

@codex_eval(20, 9, 2, 13.5)
def calc():
    print((135/360) * (6**2))
    return

show_stats(20)




# ---------------------------------------------------------
# 21, p: 12, s: 2
# PEMDAS dictates that division takes place before addition. So, 10 should be divided by 2 first. Then, the quotient should be added to 3.

@codex_eval(21, 12, 2, 8)
def calc():
    return

@codex_eval(21, 12, 2, 8)
def calc():
    return 3 + 10 / 2

@codex_eval(21, 12, 2, 8)
def calc():
    return 10 / 2 + 3

show_stats(21)




# ---------------------------------------------------------
# 22, p: 13, s: 2
# Add the digits in the tens place, 8 and 4. Since the sum exceeds 10, write down the ones of the sum (2), then carry the tens (1) to the hundreds place.

@codex_eval(22, 13, 2, 2)
def calc():
    a = 8
    b = 4
    c = a + b
    return c

@codex_eval(22, 13, 2, 2)
def calc():
    a = 8
    b = 4
    c = a + b
    print(c)

@codex_eval(22, 13, 2, 2)
def calc():
    a = 8
    b = 4
    return a + b

@codex_eval(22, 13, 2, 2)
def calc():
    a = 8
    b = 4
    sum = a + b
    return sum

@codex_eval(22, 13, 2, 2)
def calc():
    tens = 8 + 4
    return tens

@codex_eval(22, 13, 2, 2)
def calc():
    return 8 + 4

@codex_eval(22, 13, 2, 2)
def calc():
    print(8 + 4)  # 12

show_stats(22)



# ---------------------------------------------------------
# 23, p: 14, s: 2
# The slope is the quotient of the difference between the y-values 8 and 4, and the difference between the x-values 7 and 5.

@codex_eval(23, 14, 2, 2)
def calc():
    y1 = 8
    y2 = 4
    x1 = 7
    x2 = 5
    slope = (y2 - y1) / (x2 - x1)
    return slope

@codex_eval(23, 14, 2, 2)
def calc():
    x1 = 5
    y1 = 4
    x2 = 7
    y2 = 8
    slope = (y2-y1)/(x2-x1)
    return slope

@codex_eval(23, 14, 2, 2)
def calc():
    slope = (8 - 4) / (7 - 5)
    return slope

@codex_eval(23, 14, 2, 2)
def calc():
    m = (8 - 4) / (7 - 5)
    return m

@codex_eval(23, 14, 2, 2)
def calc():
    slope = (8 - 4) / (7 - 5)
    print(slope)

@codex_eval(23, 14, 2, 2)
def calc():
    y = 8 - 4
    x = 7 - 5
    result = y / x
    return result

@codex_eval(23, 14, 2, 2)
def calc():
    # Write your code here.
    return (8-4)/(7-5)

show_stats(23)


# ---------------------------------------------------------
# 24, p: 13, s: 3
# The sum of 8 and 4 without the 1. 

@codex_eval(24, 13, 3, 2)
def calc():
    return 8 + 4 - 1

show_stats(24)




# ---------------------------------------------------------
# 25, p: 2, s: 4
# Multiply 3 and 2

@codex_eval(25, 2, 4, 6)
def calc():
    return 3 * 2

show_stats(25)



# ---------------------------------------------------------
# 26, p: 3, s: 4
# Multiply 3 and 5. Multiply 2 and 4. Add those two values together

@codex_eval(26, 3, 4, 23)
def calc():
    a = 3 * 5
    b = 2 * 4
    return a + b

@codex_eval(26, 3, 4, 23)
def calc():
    return 3 * 5 + 2 * 4

@codex_eval(26, 3, 4, 23)
def calc():
    return (3 * 5) + (2 * 4)

show_stats(26)  

# ---------------------------------------------------------
# 27, p: 4, s: 4
# Multiply 4 and 5.

@codex_eval(27, 4, 4, 20)
def calc():
    return 4 * 5

show_stats(27)

# ---------------------------------------------------------
# 28, p: 6, s: 4
# Multiply 3 and 3

@codex_eval(28, 6, 4, 9)
def calc():
    return 3 * 3


show_stats(28)


# ---------------------------------------------------------
# 29, p: 7, s: 4
# Add 9 and 8

@codex_eval(29, 7, 4, 17)
def calc():
    return 9 + 8

@codex_eval(29, 7, 4, 17)
def calc():
    print(9 + 8)

show_stats(29)

# ---------------------------------------------------------
# 30, p: 8, s: 4
# The area of the larger is half of 20 times 12. The area of the smaller triangle is half of 6 times 10. Subtract the area of the smaller triangle from the area of the larger triangle.

@codex_eval(30, 8, 4, 90)
def calc():
    area1 = 0.5 * 20 * 12
    area2 = 0.5 * 6 * 10
    return area1 - area2

@codex_eval(30, 8, 4, 90)
def calc():
    area1 = 20 * 12 / 2
    area2 = 6 * 10 / 2
    return area1 - area2

@codex_eval(30, 8, 4, 90)
def calc():
    area = 20 * 12 / 2 - 6 * 10 / 2
    return area

@codex_eval(30, 8, 4, 90)
def calc():
    area = 20 * 12 / 2
    area2 = 6 * 10 / 2
    return area - area2

@codex_eval(30, 8, 4, 90)
def calc():
    return (20*12)/2 - (6*10)/2

@codex_eval(30, 8, 4, 90)
def calc():
    area1 = (1/2)*20*12
    area2 = (1/2)*6*10
    print(area1-area2)

@codex_eval(30, 8, 4, 90)
def calc():
    area = 20 * 12 / 2.0
    area = area - 6 * 10 / 2.0
    return area

@codex_eval(30, 8, 4, 90)
def calc():
    return 20*12/2 - 6*10/2

@codex_eval(30, 8, 4, 90)
def calc():
    area = (20*12)/2
    area2 = (6*10)/2
    return area-area2

@codex_eval(30, 8, 4, 90, False)
def calc():
    return 120 - (30)

show_stats(30)


# ---------------------------------------------------------
# 31, p: 10, s: 4
# The larger triangle has a ratio of 3 over 12. The smaller triangle has a ratio of 2 over x. If you set the ratios equal to each other (because they are similar triangles), you can use cross-multiplication to solve for x. Multiply 2 and 12. Divide that value by 3.
@codex_eval(31, 10, 4, 8)
def calc():
    x = (2 * 12) / 3
    print(x)

@codex_eval(31, 10, 4, 8)
def calc():
    return (2 * 12) / 3

@codex_eval(31, 10, 4, 8)
def calc():
    return 2 * 12 / 3

@codex_eval(31, 10, 4, 8)
def calc():
    return 12 * 2 / 3

@codex_eval(31, 10, 4, 8, False)
def calc():
    return 8

@codex_eval(31, 10, 4, 8, False)
def calc():
    x = 8
    return x

@codex_eval(31, 10, 4, 8)
def calc():
    return 12.0 * 2 / 3

show_stats(31)



# ---------------------------------------------------------
# 32, p: 11, s: 4
# The total percentage value is 100 plus 20. Divide 144, the number of Type A pears, by 1.2.

@codex_eval(32, 11, 4, 120)
def calc():
    return 144 / 1.2

@codex_eval(32, 11, 4, 120)
def calc():
    total = 144 / 1.2
    return total

@codex_eval(32, 11, 4, 120, False)
def calc():
    return 120

show_stats(32)

# ---------------------------------------------------------
# 33, p: 12, s: 4
# Divide 10 by 2. Then add that value to 3.

@codex_eval(33, 12, 4, 8)
def calc():
    x = 10 / 2
    y = x + 3
    return y

@codex_eval(33, 12, 4, 8)
def calc():
    x = 10
    y = 2
    z = x / y
    return z + 3

@codex_eval(33, 12, 4, 8)
def calc():
    return (10 / 2) + 3

@codex_eval(33, 12, 4, 8)
def calc():
    return 10 / 2 + 3

@codex_eval(33, 12, 4, 8)
def calc():
    return 3 + 10 / 2

show_stats(33)


# ---------------------------------------------------------
# 34, p: 1, s: 5
# Find the quotient of 12 divided by three and divide 100 by the quotient to find the percent
@codex_eval(34, 1, 5, 25)
def calc():
    quotient = 12/3
    percent = 100/quotient
    return percent

@codex_eval(34, 1, 5, 25)
def calc():
    quotient = 12 / 3
    percent = 100 / quotient
    print(percent)

@codex_eval(34, 1, 5, 25)
def calc():
    quotient = 12 / 3
    return 100 / quotient

@codex_eval(34, 1, 5, 25)
def calc():
    return 100 / (12 / 3)

@codex_eval(34, 1, 5, 25)
def calc():
    return (100 / (12 / 3))


show_stats(34)


# ---------------------------------------------------------
# 35, p: 2, s: 5
# The value of a is the product of 3 and 2. 
@codex_eval(35, 2, 5, 6)
def calc():
    a = 3 * 2
    return a

show_stats(35)




# ---------------------------------------------------------
# 36, p: 4, s: 5
# Multipy 4 and 5 to find the product that is the value for c.
@codex_eval(36, 4, 5, 20)
def calc():
    a = 4
    b = 5
    c = a * b
    return c

@codex_eval(36, 4, 5, 20)
def calc():
    c = 4 * 5
    return c


show_stats(36)

# ---------------------------------------------------------
# 37, p: 5, s: 5
# Multiply 4 x 3 and use the product 12 as your denominator.
@codex_eval(37, 5, 5, 12)
def calc():
    return 4 * 3 / 12

@codex_eval(37, 5, 5, 12)
def calc():
    return 4 * 3

@codex_eval(37, 5, 5, 12)
def calc():
    # Multiply 4 x 3 and use the product 12 as your denominator.
    return 12

@codex_eval(37, 5, 5, 12)
def calc():
    return 12/4*3

show_stats(37)


# ---------------------------------------------------------
# 38, p: 6, s: 5
# Multiply 3 by the multiple used to convert the denominator 3. 
@codex_eval(38, 6, 5, 9)
def calc():
    return 3 * 3

@codex_eval(38, 6, 5, 9)
def calc():
    return 3 * 1

@codex_eval(38, 6, 5, 9)
def calc():
    return 3 * (1/3)

@codex_eval(38, 6, 5, 9)
def calc():
    # The multiple used to convert the denominator 3 is 2.
    return 3 * 2

show_stats(38)


# ---------------------------------------------------------
# 39, p: 7, s: 5
# Add 9 plus 8. 
@codex_eval(39, 7, 5, 17)
def calc():
    return 9 + 8


show_stats(39)

# ---------------------------------------------------------
# 40, p: 8, s: 5
# Multiply 12 and 20 and divide by two.  Then multiply 10 and 6 and divide that product by two.  Then subtract the smaller triangle area from the larger triangle area. 
@codex_eval(40, 8, 5, 90)
def calc():
    return (12 * 20 / 2) - (10 * 6 / 2)

@codex_eval(40, 8, 5, 90)
def calc():
    return (12 * 20) / 2 - (10 * 6) / 2

@codex_eval(40, 8, 5, 90)
def calc():
    # Calculate the area of the larger triangle.
    larger_triangle = (12 * 20) / 2
    # Calculate the area of the smaller triangle.
    smaller_triangle = (10 * 6) / 2
    # Subtract the smaller triangle area from the larger triangle area.
    return larger_triangle - smaller_triangle

@codex_eval(40, 8, 5, 90, False)
def calc():
    return 120

show_stats(40)


# ---------------------------------------------------------
# 41, p: 10, s: 5
# Divide 3 by 2 and use the quotient to divide 12 by. 
@codex_eval(41, 10, 5, 8)
def calc():
    quotient = 3 / 2
    return 12 / quotient

@codex_eval(41, 10, 5, 8)
def calc():
    return 12 / (3 / 2)

@codex_eval(41, 10, 5, 8)
def calc():
    a = 3
    b = 2
    c = 12
    d = a/b
    return c/d

@codex_eval(41, 10, 5, 8)
def calc():
    x = 3
    y = 12
    z = y // x
    return z

@codex_eval(41, 10, 5, 8)
def calc():
    return 12 / 3 / 2

@codex_eval(41, 10, 5, 8)
def calc():
    print(12 / (3 / 2))


show_stats(41)

# ---------------------------------------------------------
# 42, p: 12, s: 5
# Divide 10 by 2 then add 3. 

@codex_eval(42, 12, 5, 8)
def calc():
    return (10 / 2) + 3

@codex_eval(42, 12, 5, 8)
def calc():
    return 10 / 2 + 3

show_stats(42)


# ---------------------------------------------------------
# 43, p: 14, s: 5
# subtract 7 and 5, then divide the product by 8 minus 4
@codex_eval(43, 14, 5, 2)
def calc():
    return (7 - 5) / (8 - 4)

@codex_eval(43, 14, 5, 2)
def calc():
    return ((7 - 5) / (8 - 4))

show_stats(43)



# ---------------------------------------------------------
# 44, p: 1, s: 6
# divide 3 by 12 to get .25, then multiply by 100 which is 25 percent.
@codex_eval(44, 1, 6, 25)
def calc():
    return (3 / 12) * 100

@codex_eval(44, 1, 6, 25)
def calc():
    return 3 / 12 * 100

@codex_eval(44, 1, 6, 25)
def calc():
    return (3.0/12.0)*100

show_stats(44)

# ---------------------------------------------------------
# 45, p: 2, s: 6
# To find a you would use the foil method and multiply the first numbers in each set, which are 3x and 2x which would give you 6x² so a is 6.
@codex_eval(45, 2, 6, 6)
def calc():
    a = 6
    b = 3
    c = 2
    x = 1
    return a * x**2 + b * x + c

@codex_eval(45, 2, 6, 6)
def calc():
    a = 6
    b = 3
    c = 2
    d = 1
    return a * b * c * d

@codex_eval(45, 2, 6, 6, False)
def calc():
    return 6

@codex_eval(45, 2, 6, 6, False)
def calc():
    # Your code here
    return 6

show_stats(45)

# ---------------------------------------------------------
# 46, p: 3, s: 6
# Using foil and multiply the inside and outside values (4 times 2x, which is 8x) and (3x times 5 which is 15x) and then adding these together we get 23x so the value for b is 23.
@codex_eval(46, 3, 6, 23)
def calc():
  a = 4
  b = 2
  c = 3
  d = 5
  return (a * b) + (c * d)

@codex_eval(46, 3, 6, 23, False)
def calc():
    return 23

@codex_eval(46, 3, 6, 23, False)
def calc():
    # Write your code here
    return 23

@codex_eval(46, 3, 6, 23)
def calc():
    return 8 * 15 + 23

show_stats(46)

# ---------------------------------------------------------
# 47, p: 4, s: 6
# You multiply 4 times 5 and get 20 and that's the answer because they have no variables they can't be added to any other numbers.
@codex_eval(47, 4, 6, 20)
def calc():
    return 4 * 5

@codex_eval(47, 4, 6, 20, False)
def calc():
    return 20

@codex_eval(47, 4, 6, 20)
def calc():
    print(4*5)

show_stats(47)


# ---------------------------------------------------------
# 48, p: 6, s: 6
# To get a denominator of 12 from 4, you multiply by 3 on the bottom, so you have to do this on the top as well. 3 times 3 equals 9. three-fourths equals nine-twelfths.
@codex_eval(48, 6, 6, 9)
def calc():
    a = 3
    b = 4
    c = 9
    d = 12
    return a * b / c * d

@codex_eval(48, 6, 6, 9, False)
def calc():
    return 9

@codex_eval(48, 6, 6, 9)
def calc():
    return 9/12

@codex_eval(48, 6, 6, 9)
def calc():
    return 4 * 3 / 4 * 3

@codex_eval(48, 6, 6, 9)
def calc():
    return 3 * 3

show_stats(48)

# ---------------------------------------------------------
# 49, p: 7, s: 6
# 9 plus 8 equals 17
@codex_eval(49, 7, 6, 17)
def calc():
    a = 9
    b = 8
    return a + b

@codex_eval(49, 7, 6, 17)
def calc():
    return 9 + 8

show_stats(49)

# ---------------------------------------------------------
# 50, p: 8, s: 6
# The area for the larger triangle is (one half times 20 km times 12 km which equals 120km squared) the smaller triangle's area is (one half times 6 km times 10 km which equals 30 km squared.) Then you subtract the small white area to leave the shade region which is 120 km squared minus 30 km squared.
@codex_eval(50, 8, 6, 90)
def calc():
    # This is the area of the larger triangle
    area1 = (1/2) * 20 * 12
    # This is the area of the smaller triangle
    area2 = (1/2) * 6 * 10
    # This is the area of the small white region
    area3 = 1/2 * 6 * 2
    # This is the area of the shaded region
    area4 = area1 - area2 - area3
    return area4

@codex_eval(50, 8, 6, 90)
def calc():
    area = (1/2)*20*12 - (1/2)*6*10

@codex_eval(50, 8, 6, 90)
def calc():
    """This function calculates the area of the shaded region in a triangle."""
    bigArea = (1/2) * 20 * 12
    smallArea = (1/2) * 6 * 10
    shadeArea = bigArea - smallArea
    print("The area of the shaded region is", shadeArea, "km squared.")

@codex_eval(50, 8, 6, 90)
def calc():
    return 120 - 30

@codex_eval(50, 8, 6, 90)
def calc():
    area = 120 - 30
    print(area)
    return area

@codex_eval(50, 8, 6, 90)
def calc():
    return (120-30)

show_stats(50)

# ---------------------------------------------------------
# 51, p: 9, s: 6
# Using the formula and plugging in our info we get 135 over 360 times pi 6 squared which equals  13.5 pi cm squared
import math
pi = 3.14
@codex_eval(51, 9, 6, 13.5*pi)
def calc():
    pi = 3.14
    radius = 3
    height = 6
    volume = pi * radius ** 2 * height
    surface_area = 2 * pi * radius * height + 2 * pi * radius ** 2
    return surface_area

@codex_eval(51, 9, 6, 13.5*math.pi)
def calc():
    return 135 / (360 * math.pi * 6 ** 2)

@codex_eval(51, 9, 6, 13.5*pi)
def calc():
    print(135/360*3.14*6**2)

@codex_eval(51, 9, 6, 13.5*math.pi)
def calc():
    return (135 / (360 * math.pi * 6 ** 2))

@codex_eval(51, 9, 6, 13.5*math.pi)
def calc():
    return (135/360) * (6**2) * math.pi

@codex_eval(51, 9, 6, 13.5*pi)
def calc():
    r = 6
    h = 135
    pi = 3.14
    print((h/(2*pi*r))*pi*r**2)

@codex_eval(51, 9, 6, 13.5*math.pi)
def calc():
    return 135 * 6**2 * math.pi

@codex_eval(51, 9, 6, 13.5*math.pi)
def calc():
    print("The area of the circle is", 135 / (360 * (math.pi * 6 ** 2)), "pi cm squared")

@codex_eval(51, 9, 6, 13.5*pi)
def calc():
    return (135/360)*(3.14*6**2)

show_stats(51)

# ---------------------------------------------------------
# 52, p: 10, s: 6
# 12 divided by 3 is 4 so the green triangle's longer side is 4 times larger than its smaller one. Now you take the 2 on the smaller side of the pink triangle and multiply it by 4 to get 8 yd. 
@codex_eval(52, 10, 6, 8, False)
def calc():
    print("12 divided by 3 is 4 so the green triangle's longer side is 4 times larger than its smaller one. Now you take the 2 on the smaller side of the pink triangle and multiply it by 4 to get 8 yd.")
    return 8

@codex_eval(52, 10, 6, 8)
def calc():
    print("12/3 = 4")
    print("4 * 2 = 8")
    print("Answer: 8")

@codex_eval(52, 10, 6, 8)
def calc():
    return 12/3*2

@codex_eval(52, 10, 6, 8)
def calc():
    print(12/3*4*2)

@codex_eval(52, 10, 6, 8)
def calc():
    return 12/3*4*2

@codex_eval(52, 10, 6, 8, False)
def calc():
    return 8

@codex_eval(52, 10, 6, 8)
def calc():
    print(12/3 * 4)

@codex_eval(52, 10, 6, 8)
def calc():
    return 12/3

show_stats(52)


# ---------------------------------------------------------
# 53, p: 11, s: 6
# a equals b plus 20 percent so 120 percent. b equals 100 percent. 100 divided by 120 equals 83.33 percent. So b produces 83.33 percent (.8333) of a (144) which equals 120
@codex_eval(53, 11, 6, 120)
def calc():
    a = 144
    b = a * .8333
    return b

@codex_eval(53, 11, 6, 120)
def calc():
    a = 144
    b = a * 0.8333
    return b

@codex_eval(53, 11, 6, 120)
def calc():
    a = 144
    b = a * 0.8333
    print(b)
    return b

@codex_eval(53, 11, 6, 120)
def calc():
    a = 144
    b = 100
    return a * (b / (b + (b * .2)))

@codex_eval(53, 11, 6, 120)
def calc():
    a = 144
    b = 100
    return a / (b + (b * 0.2))

@codex_eval(53, 11, 6, 120)
def calc():
    a = 144
    b = a / 1.2
    print(b)
    print(a / b)  # 120

show_stats(53)

# ---------------------------------------------------------
# 54, p: 12, s: 6
# divide 10 by 2 then add 3
@codex_eval(54, 12, 6, 8)
def calc():
    return 10 / 2 + 3

@codex_eval(54, 12, 6, 8)
def calc():
    return (10 / 2) + 3


show_stats(54)


# ---------------------------------------------------------
# 55, p: 13, s: 6
# add 8 to 4 which gives you 12 and you put down the ones column, which is a two and carry the one over to the next column to the left.
@codex_eval(55, 13, 6, 2)
def calc():
    x = 8
    y = 4
    z = x + y
    ones = z % 10
    tens = z // 10
    print(ones)
    print(tens)

@codex_eval(55, 13, 6, 2)
def calc():
    a = 8
    b = 4
    c = a + b
    return c

@codex_eval(55, 13, 6, 2)
def calc():
    x = 8
    y = 4
    z = x + y
    return z

@codex_eval(55, 13, 6, 2)
def calc():
    x = 4
    y = 8
    z = x + y
    print(z)
    return z

@codex_eval(55, 13, 6, 2)
def calc():
    return 8 + 4

@codex_eval(55, 13, 6, 2)
def calc():
    # 8 + 4 = 12
    # 1 + 2 = 3
    return 3

@codex_eval(55, 13, 6, 2)
def calc():
    print(4 + 8)

@codex_eval(55, 13, 6, 2)
def calc():
    return 4 + 8

@codex_eval(55, 13, 6, 2)
def calc():
    return 12

show_stats(55)

# ---------------------------------------------------------
# 56, p: 14, s: 6
# You take y2 which is 8 and subtract y1 which is 4 and get 4. Put that aside for a minute and take x2 (7) and subtract x1 (5) which is 2. Take the four from the differences in the y values and put that over the two from the difference of x values and you get 4 over 2 which is 2.
@codex_eval(56, 14, 6, 2)
def calc():
    y2 = 8
    y1 = 4
    x2 = 7
    x1 = 5
    return (y2 - y1) / (x2 - x1)

@codex_eval(56, 14, 6, 2)
def calc():
    x1 = 5
    y1 = 4
    x2 = 7
    y2 = 8
    return (y2 - y1)/(x2 - x1)

@codex_eval(56, 14, 6, 2)
def calc():
    x = (8-4)/(7-5)
    return x

@codex_eval(56, 14, 6, 2)
def calc():
    return 2

@codex_eval(56, 14, 6, 2)
def calc():
    # Your code here
    return 2

show_stats(56)

# ---------------------------------------------------------
# 57, p: 1, s: 7
# Multiply 3 times 100. Then, divide the product by 12 to get the percentage.
@codex_eval(57, 1, 7, 25)
def calc():
    return (3 * 100) / 12

@codex_eval(57, 1, 7, 25)
def calc():
    return 3 * 100 / 12


show_stats(57)


# ---------------------------------------------------------
# 58, p: 2, s: 7
# Multiply 3 and 2 together because they are both combined with x. Write the product as the value of a.
@codex_eval(58, 2, 7, 6)
def calc():
    x = 3
    y = 2
    a = x * y
    return a

@codex_eval(58, 2, 7, 6)
def calc():
    x = 2
    y = 3
    a = x * y
    return a

@codex_eval(58, 2, 7, 6)
def calc():
    a = 3
    b = 2
    c = a * b
    return c

@codex_eval(58, 2, 7, 6)
def calc():
    a = 3 * 2
    return a

show_stats(58)



# ---------------------------------------------------------
# 59, p: 3, s: 7
# Multiply the 3 of the 3x by the 5. Multiply the 2 of the 2x by the 4. Add each of those products together to fine the value of b.
@codex_eval(59, 3, 7, 23)
def calc():
    a = 3
    b = 5
    c = 2
    d = 4
    e = a * b
    f = c * d
    g = e + f
    return g

@codex_eval(59, 3, 7, 23)
def calc():
    b = 3 * 5 + 2 * 4
    return b

@codex_eval(59, 3, 7, 23)
def calc():
    x = 3
    y = 5
    a = x * y
    x = 2
    y = 4
    b = x * y
    c = a + b
    return c

@codex_eval(59, 3, 7, 23)
def calc():
    return 3 * 5 + 2 * 4

@codex_eval(59, 3, 7, 23)
def calc():
    b = (3*5) + (2*4)
    return b

@codex_eval(59, 3, 7, 23)
def calc():
    a = 3*5
    b = 2*4
    c = a+b
    return c

@codex_eval(59, 3, 7, 23)
def calc():
    a = 3*5
    b = 2*4
    return a + b

@codex_eval(59, 3, 7, 23)
def calc():
    b = 5*3 + 4*2
    return b

show_stats(59)

# ---------------------------------------------------------
# 60, p: 4, s: 7
# Multiply 4 times 5 to find the value of c.
@codex_eval(60, 4, 7, 20)
def calc():
    a = 4
    b = 5
    c = a * b
    return c

@codex_eval(60, 4, 7, 20)
def calc():
    a = 4
    b = 5
    c = a * b
    print(c)

@codex_eval(60, 4, 7, 20)
def calc():
    c = 4 * 5
    return c

show_stats(60)



# ---------------------------------------------------------
# 61, p: 6, s: 7
# Your denominator 4 goes into 12 3 times, so multiply your numerator, 3, by 3 as well. The product is your new numerator.
@codex_eval(61, 6, 7, 9)
def calc():
    numerator = 3
    denominator = 4
    quotient = 12 / denominator
    new_numerator = numerator * quotient
    return new_numerator

@codex_eval(61, 6, 7, 9, False)
def calc():
    return 9

@codex_eval(61, 6, 7, 9)
def calc():
    return 3*3

@codex_eval(61, 6, 7, 9)
def calc():
    return 12/4*3

show_stats(61)

# ---------------------------------------------------------
# 62, p: 7, s: 7
# Add the numerator 9, to the numerator 8. The sum is the numerator of the new fraction.
@codex_eval(62, 7, 7, 17)
def calc():
    return 9 + 8

@codex_eval(62, 7, 7, 17, False)
def calc():
    return 17

show_stats(62)

# ---------------------------------------------------------
# 63, p: 10, s: 7
# Use the length of the two similar sides to make a ratio. In this case, use 2 and 3 yards to make the ratio, 2/3. Make the ratio with the other two similar sides, in this case x and 12, to make the ratio x/12. Cross-multiply the two ratios and solve for x. 
@codex_eval(63, 10, 7, 8)
def calc():
    x = 2 * 12 / 3
    print(x)

@codex_eval(63, 10, 7, 8)
def calc():
    x = 12 * 3 / 2
    return x

@codex_eval(63, 10, 7, 8)
def calc():
    x = 2*12/3
    return x

@codex_eval(63, 10, 7, 8)
def calc():
    # Get the length of the two similar sides.
    x = 2
    y = 3
    # Make the ratio.
    ratio1 = x/y
    # Get the length of the other two similar sides.
    x = 12
    y = x/ratio1
    # Print the length of the other two similar sides.
    print(y)

@codex_eval(63, 10, 7, 8)
def calc():
    x = 6
    y = 12
    z = 6
    print(x / y * z)

@codex_eval(63, 10, 7, 8)
def calc():
    x = 24/3
    print(x)
show_stats(63)

# ---------------------------------------------------------
# 64, p: 12, s: 7
# Divide 10 by 2. Then add the answer to 3.
@codex_eval(64, 12, 7, 8)
def calc():
    a = 10 / 2
    b = a + 3
    return b

@codex_eval(64, 12, 7, 8)
def calc():
    return (10 / 2) + 3

@codex_eval(64, 12, 7, 8)
def calc():
    return 10 / 2 + 3

@codex_eval(64, 12, 7, 8)
def calc():
    answer = 10/2 + 3
    return answer

@codex_eval(64, 12, 7, 8)
def calc():
    return 5 + 3

show_stats(64)

# ---------------------------------------------------------
# 65, p: 13, s: 7
# Add 8 plus 4. Bring down the answer's ones place digit, next to the 9. Carry the tens place digit up above the 3 to be added in the hundreds place.
@codex_eval(65, 13, 7, 2)
def calc():
    a = 8
    b = 4
    c = 9
    d = 3
    return a + b

@codex_eval(65, 13, 7, 2)
def calc():
    return 8 + 4

@codex_eval(65, 13, 7, 2)
def calc():
    return 8 + 4 + 9 + 3

@codex_eval(65, 13, 7, 2)
def calc():
    print("8 + 4 =", 8 + 4)
    print("9 + 3 =", 9 + 3)

@codex_eval(65, 13, 7, 2)
def calc():
    a = 8 + 4
    return a

@codex_eval(65, 13, 7, 2)
def calc():
    return 12

@codex_eval(65, 13, 7, 2)
def calc():
    return 8 + 4
    return 9 + 3

show_stats(65)


# ---------------------------------------------------------
# 66, p: 14, s: 7
# The change in our x points if found by subtracting 5 from 7. The change in our y points is found by subtracting 4 from 8. Divide the change in y by the change x to find the slope.
@codex_eval(66, 14, 7, 2)
def calc():
    x1 = 5
    x2 = 7
    y1 = 4
    y2 = 8
    slope = (y2 - y1) / (x2 - x1)
    return slope

@codex_eval(66, 14, 7, 2)
def calc():
    x1 = 5
    y1 = 4
    x2 = 7
    y2 = 8
    change_x = x2 - x1
    change_y = y2 - y1
    slope = change_y / change_x
    return slope

@codex_eval(66, 14, 7, 2)
def calc():
    x1 = 7
    x2 = 5
    y1 = 8
    y2 = 4
    slope = (y2 - y1) / (x2 - x1)
    return (slope)

@codex_eval(66, 14, 7, 2)
def calc():
    x = 7 - 5
    y = 8 - 4
    slope = y / x
    return (slope)

@codex_eval(66, 14, 7, 2)
def calc():
    x = 7 - 5
    y = 8 - 4
    slope = y / x
    return slope

@codex_eval(66, 14, 7, 2)
def calc():
    x1 = 5
    y1 = 4
    x2 = 7
    y2 = 8
    slope = (y2-y1)/(x2-x1)
    return (slope)

@codex_eval(66, 14, 7, 2)
def calc():
    y1 = 8
    y2 = 4
    x1 = 7
    x2 = 5
    slope = (y1 - y2) / (x1 - x2)
    return (slope)

@codex_eval(66, 14, 7, 2)
def calc():
    return (8 - 4) / (7 - 5)

@codex_eval(66, 14, 7, 2)
def calc():
    x = 7 - 5
    y = 8 - 4
    return (y/x)

@codex_eval(66, 14, 7, 2)
def calc():
    return (8 - 4) / (7 - 5)

show_stats(66)


# ---------------------------------------------------------
# 67, p: 1, s: 8
# Multiply the numerator 3 by 100 and divide that result by the numerator 12 to find the percentage that 3 is of 12.
@codex_eval(67, 1, 8, 25)
def calc():
    numerator = 3
    denominator = 12
    return (numerator * 100) / denominator

@codex_eval(67, 1, 8, 25)
def calc():
    return (3 * 100) / 12

@codex_eval(67, 1, 8, 25)
def calc():
    return 3 * 100 / 12

@codex_eval(67, 1, 8, 25)
def calc():
    print(100 * 3 / 12)
    return

show_stats(67)



# ---------------------------------------------------------
# 68, p: 2, s: 8
# Multiply the numerator 3 by the numerator 2 to find the value of a.
@codex_eval(68, 2, 8, 6)
def calc():
    a = 3 * 2
    return a

@codex_eval(68, 2, 8, 6)
def calc():
    a = 3 * 2
    # Return the value of a.
    return a


show_stats(68)



# ---------------------------------------------------------
# 69, p: 3, s: 8
# Multiply the numerator 3 by the numerator 5, then multiply the numerator 2 by the numerator 4, and then add the results to find the value of b.
@codex_eval(69, 3, 8, 23)
def calc():
    a = 3 * 5
    b = 2 * 4 + a
    return b

@codex_eval(69, 3, 8, 23)
def calc():
    a = 3
    b = 5
    c = 2
    d = 4
    return a * b + c * d

@codex_eval(69, 3, 8, 23)
def calc():
    a = 3 * 5
    b = 2 * 4 + a
    print(b)

@codex_eval(69, 3, 8, 23)
def calc():
    a = 3 * 5
    b = 2 * 4
    return a + b

show_stats(69)


# ---------------------------------------------------------
# 70, p: 4, s: 8
# Multiply the numerator 4 by the numerator 5 to find the value of c.
@codex_eval(70, 4, 8, 20)
def calc():
    a = 4
    b = 5
    c = a * b
    return c

@codex_eval(70, 4, 8, 20)
def calc():
    a = 4
    b = 5
    c = a * b
    print(c)

@codex_eval(70, 4, 8, 20)
def calc():
    a = 4
    b = 5
    c = a * b
    print(c)
    return c

@codex_eval(70, 4, 8, 20)
def calc():
    c = 4 * 5
    return c

show_stats(70)


# ---------------------------------------------------------
# 71, p: 5, s: 8
# Multiply the denominator 4 by the denominator 3 to find the common denominator to be used for the converted fractions.
@codex_eval(71, 5, 8, 12)
def calc():
    return 4*3

@codex_eval(71, 5, 8, 12, False)
def calc():
    return 12

@codex_eval(71, 5, 8, 12)
def calc():
    return (4*3)


show_stats(71)

# ---------------------------------------------------------
# 72, p: 6, s: 8
# Multiply the numerator 3 by the denominator 3 to find the numerator of the left converted fraction.
@codex_eval(72, 6, 8, 9)
def calc():
    num = 3
    den = 3
    num = num * den
    return num

@codex_eval(72, 6, 8, 9)
def calc():
    return 3 * 3

@codex_eval(72, 6, 8, 9, False)
def calc():
    return 9

@codex_eval(72, 6, 8, 9)
def calc():
    return (3 * 3) / 3


# ---------------------------------------------------------
# 73, p: 7, s: 8
# Sum the numerator 9 and the numerator 8 to find the numerator of the simplified fraction.
@codex_eval(73, 7, 8, 17)
def calc():
    numerator = 9 + 8
    return numerator

@codex_eval(73, 7, 8, 17)
def calc():
    num = 9 + 8
    return num

@codex_eval(73, 7, 8, 17)
def calc():
    numerator = 9 + 8
    denominator = 7
    return (numerator, denominator)

@codex_eval(73, 7, 8, 17)
def calc():
    numerator = 9 + 8
    denominator = 15
    return numerator, denominator

@codex_eval(73, 7, 8, 17)
def calc():
    numerator = 9 + 8
    denominator = 3
    return numerator, denominator

@codex_eval(73, 7, 8, 17)
def calc():
    return 9 + 8

@codex_eval(73, 7, 8, 17)
def calc():
    return (9 + 8)


show_stats(73)

# ---------------------------------------------------------
# 74, p: 8, s: 8
# Find the area of the larger triangle by multiplying the base (20) by the width (12) and dividing that in half. Then do the same for the smaller triangle by multiplying the base (10) by the width (6) and dividing in half. Here, the rectangles have areas of 120 and 30. To find the area of the shaded region, subtract the area 30 from the area 120.
@codex_eval(74, 8, 8, 90)
def calc():
    return

show_stats(74)

@codex_eval(74, 8, 8, 90)
def calc():
    # Calculate the area of the larger triangle
    large_triangle = (20 * 12) / 2
    # Calculate the area of the smaller triangle
    small_triangle = (10 * 6) / 2
    # Calculate the area of the rectangle
    rectangle = large_triangle - small_triangle
    # Print the result


@codex_eval(74, 8, 8, 90)
def calc():
    area1 = (20*12)/2
    area2 = (10*6)/2
    return area1 - area2

@codex_eval(74, 8, 8, 90)
def calc():
    # Calculate the area of the larger triangle
    large = (20 * 12) / 2
    # Calculate the area of the smaller triangle
    small = (10 * 6) / 2
    # Calculate the area of the shaded region
    area = large - small
    return area

@codex_eval(74, 8, 8, 90)
def calc():
    return 120 - 30

@codex_eval(74, 8, 8, 90)
def calc():
    x = 120 - 30
    return x

show_stats(74)

# ---------------------------------------------------------
# 75, p: 9, s: 8
# First, find the area of the circle by multiplying the radius 6 by itself, as in 6 times 6. This multiplied by pi (3.14) gives you the area of the circle. Then figure out the percentage of the area covered by the sector by dividing the degrees of the arc (135) by the degrees of the whole circle (360). Finally, to find the area of the sector, multiply the area of the circle (113.04) by the decimal representing the percentage covered by the sector (0.375) to find the area of the sector (42.39).
@codex_eval(75, 9, 8, 42.39)
def calc():
    pass

@codex_eval(75, 9, 8, 42.39)
def calc():
    pi = 3.14
    radius = 6
    degrees_of_arc = 135
    degrees_of_whole_circle = 360
    area_of_circle = pi * (radius * radius)
    percentage_of_area_covered_by_sector = degrees_of_arc / degrees_of_whole_circle
    area_of_sector = area_of_circle * percentage_of_area_covered_by_sector

@codex_eval(75, 9, 8, 42.39)
def calc():
    pi = 3.14
    radius = 6
    degrees_of_circle = 360
    degrees_of_arc = 135
    area_of_circle = radius * radius * pi
    percentage_of_area = degrees_of_arc / degrees_of_circle
    area_of_sector = area_of_circle * percentage_of_area

@codex_eval(75, 9, 8, 42.39)
def calc():
    pi = 3.14
    radius = 6
    area_of_circle = pi * (radius**2)
    area_of_sector = area_of_circle * 0.375
    return area_of_sector

@codex_eval(75, 9, 8, 42.39)
def calc():
    pi = 3.14
    r = 6
    area = pi * r * r
    percent = 135 / 360
    area_sector = area * percent
    return area_sector

@codex_eval(75, 9, 8, 42.39)
def calc():
    area = 6*6*3.14
    percentage = 135/360
    sector = area*percentage
    return sector

@codex_eval(75, 9, 8, 42.39)
def calc():
    pi = 3.14
    r = 6
    degrees = 135
    degrees_circle = 360
    area_circle = pi * (r**2)
    percentage = float(degrees) / float(degrees_circle)
    area_sector = area_circle * percentage
    return area_sector

@codex_eval(75, 9, 8, 42.39)
def calc():
    pi = 3.14
    r = 6
    d = 360
    a = 135
    area = pi * r * r
    area_sector = area * (a / d)
    return area_sector

@codex_eval(75, 9, 8, 42.39, False)
def calc():
    return 42.39

show_stats(75)

# ---------------------------------------------------------
# 76, p: 11, s: 8
# When something is 20 percent more than something else, you can put that in percentage form by dividing the percent by 100 and then adding 1. Here, that would result in the number 1.2. To find the number of pears produced by Type B trees, divide the number 144 by the decimal representing the percentage (1.2).
@codex_eval(76, 11, 8, 120)
def calc():
    return 144/1.2

@codex_eval(76, 11, 8, 120, False)
def calc():
    return 120

show_stats(76)

# ---------------------------------------------------------
# 77, p: 12, s: 8
# First divide the numerator 10 by the numerator 2, because division comes before addition. Then sum that result (5) and the numerator 3 to find the final answer.
@codex_eval(77, 12, 8, 8)
def calc():
    return (10 / 2) + 3

@codex_eval(77, 12, 8, 8)
def calc():
    return 10 / 2 + 3

show_stats(77)


# ---------------------------------------------------------
# 78, p: 13, s: 8
# Add the numeral 8 plus the numeral 4, and then use the last digit of the resulting sum (12) to find the missing value (2).
@codex_eval(78, 13, 8, 2)
def calc():
    return 8 + 4

@codex_eval(78, 13, 8, 2)
def calc():
    return 8 + 4 - 2

@codex_eval(78, 13, 8, 2)
def calc():
    return 8 + 4 - 12

@codex_eval(78, 13, 8, 2, False)
def calc():
    return 2

@codex_eval(78, 13, 8, 2)
def calc():
    return 8 + 4 - 12 % 10

@codex_eval(78, 13, 8, 2)
def calc():
    return 8 + 4 % 10


show_stats(78)

# ---------------------------------------------------------
# 79, p: 14, s: 8
# Subtract the first y-coordinate 4 from the second y-coordinate 8, then subtract the first x-coordinate 5 from the second x-coordinate 7, and then divide the difference of the y-coordinates (4) by the difference of the x-coordinates (2).
@codex_eval(79, 14, 8, 2)
def calc():
    y2 = 8
    y1 = 4
    x2 = 7
    x1 = 5
    y = y2 - y1
    x = x2 - x1
    return y / x

@codex_eval(79, 14, 8, 2)
def calc():
    y = 8 - 4
    x = 7 - 5
    return

@codex_eval(79, 14, 8, 2)
def calc():
    x1 = 5
    y1 = 4
    x2 = 7
    y2 = 8
    print((y2 - y1) / (x2 - x1))

@codex_eval(79, 14, 8, 2)
def calc():
    return (8 - 4) / (7 - 5)

@codex_eval(79, 14, 8, 2)
def calc():
    y = 8 - 4
    x = 7 - 5
    return y / x

show_stats(79)


# ---------------------------------------------------------
# 80, p: 2, s: 9
# In order to find the value of a in the polynomial, the variables 3x and 2x will be multiplied together
@codex_eval(80, 2, 9, 6)
def calc():
    x = 3
    y = 2
    z = x * y
    return z

@codex_eval(80, 2, 9, 6)
def calc():
    a = 3 * 2
    return a

@codex_eval(80, 2, 9, 6, False)
def calc():
    return 6

@codex_eval(80, 2, 9, 6)
def calc():
    return 3 * 2

show_stats(80)


# ---------------------------------------------------------
# 81, p: 4, s: 9
# To get the value of c, 4 and 5 will be multiplied together
@codex_eval(81, 4, 9, 20)
def calc():
    a = 4
    b = 5
    c = a * b
    return c

@codex_eval(81, 4, 9, 20)
def calc():
    a = 2
    b = 3
    c = a * b
    return c

@codex_eval(81, 4, 9, 20)
def calc():
    a = 1
    b = 2
    c = 3
    return a + b + c

@codex_eval(81, 4, 9, 20)
def calc():
    return 4 * 5

show_stats(81)


# ---------------------------------------------------------
# 82, p: 5, s: 9
# In order to simplify the fractions, 3/4 will be multiplied by 3 to get 9/12. 2/3 will be multiplied by 4 to get 8/12. Now that the denominator of both fractions is 12, they are ready to be simplified.
@codex_eval(82, 5, 9, 12)
def calc():
    return 9/12 + 8/12

@codex_eval(82, 5, 9, 12)
def calc():
    return 9/12 * 8/12

@codex_eval(82, 5, 9, 12)
def calc():
    # Write your code here
    return 0

show_stats(82)

# ---------------------------------------------------------
# 83, p: 6, s: 9
# To find the numerator of the left converted fraction, multiply the left unconverted fraction, 3/4, by the variable 3. This will multiply with the unconverted numerator of 3 to result in the converted numerator of 9.
@codex_eval(83, 6, 9, 9, False)
def calc():
    # Path: foo.py
    # To find the numerator of the left converted fraction, multiply the left unconverted fraction, 3/4, by the variable 3. This will multiply with the unconverted numerator of 3 to result in the converted numerator of 9.
    return 9

@codex_eval(83, 6, 9, 9)
def calc():
    numerator = 3 * 3
    denominator = 4
    return numerator, denominator

@codex_eval(83, 6, 9, 9)
def calc():
    print(3 * 3 / 4)

@codex_eval(83, 6, 9, 9)
def calc():
    return 3 * 3 / 4

@codex_eval(83, 6, 9, 9, False)
def calc():
    return 9

show_stats(83)

# ---------------------------------------------------------
# 84, p: 7, s: 9
# To find the numerator of the simplified fraction, add the numerators from the converted fractions, 9 and 8, together to get 17.
@codex_eval(84, 7, 9, 17)
def calc():
    print("To find the numerator of the simplified fraction, add the numerators from the converted fractions, 9 and 8, together to get 17.")
    return 17

@codex_eval(84, 7, 9, 17)
def calc():
    a = 9
    b = 8
    return a + b

@codex_eval(84, 7, 9, 17)
def calc():
    return 17

show_stats(84)

# ---------------------------------------------------------
# 85, p: 10, s: 9
# To find the length of side x, divide 3 by 2 to get 1.5. Take 1.5 and multiply it with 12
@codex_eval(85, 10, 9, 8)
def calc():
    x = 3 / 2 * 12
    return x

@codex_eval(85, 10, 9, 8)
def calc():
    x = 12 * (3 / 2)
    return x

@codex_eval(85, 10, 9, 8)
def calc():
    return 1.5 * 12

@codex_eval(85, 10, 9, 8)
def calc():
    x = 12 * 1.5
    return x

@codex_eval(85, 10, 9, 8)
def calc():
    x = 12 / 2 * 3
    return x

@codex_eval(85, 10, 9, 8)
def calc():
    x = 3/2 * 12
    print(x)

@codex_eval(85, 10, 9, 8)
def calc():
    x = 12 * (3 / 2)
    print(x)

show_stats(85)



# ---------------------------------------------------------
# 86, p: 12, s: 9
# Take 10 divided by 2 to get 5, and then find the sum of 5 and 3
@codex_eval(86, 12, 9, 8)
def calc():
    return (10 / 2) + 3

@codex_eval(86, 12, 9, 8)
def calc():
    return 10 / 2 + 3

@codex_eval(86, 12, 9, 8)
def calc():
    return 5 + 3

@codex_eval(86, 12, 9, 8)
def calc():
    10 / 2 + 3

show_stats(86)

# ---------------------------------------------------------
# 87, p: 14, s: 9
# Take 4 and subtract it from 8 to get 4. Take 5 and subtract it from 7. Take 4 divided by 2 to get 2
@codex_eval(87, 14, 9, 2)
def calc():
    return 4 - 8 + 5 - 7 + 4 / 2

@codex_eval(87, 14, 9, 2)
def calc():
    return 4 - 8 + 7 - 5 + 4 / 2

@codex_eval(87, 14, 9, 2)
def calc():
    return 4 - 8 - 5 + 7 - 4 / 2

@codex_eval(87, 14, 9, 2)
def calc():
    return 4 - 8 - (7 - 5) - (4 / 2)

@codex_eval(87, 14, 9, 2)
def calc():
    return 4

@codex_eval(87, 14, 9, 2)
def calc():
    # Write your code here
    pass

@codex_eval(87, 14, 9, 2)
def calc():
    # write your code here
    return 0

@codex_eval(87, 14, 9, 2)
def calc():
    # Your code here
    return 0

show_stats(87)


# -------------------------------------------------------------------------------
# : Summary Statistics

import numpy as np

# cdx_noex_good_has_corr_arr = np.ones((14,10))*np.nan
cdx_good_has_corr_arr = np.ones((10,14))*np.nan

cdx_noex_good_n_incorr_arr = np.ones((10,14))*np.nan
cdx_good_n_incorr_arr = np.ones((10,14))*np.nan

cdx_noex_good_only_corr_arr = np.ones((10,14))*np.nan
cdx_good_only_corr_arr = np.ones((10,14))*np.nan

c = 0
for data in stats.values():
    i = data['student']
    j = data['problem']-1

    has_correct =  int(data['fn_correct'] > 0)
    noex_n_incorrect = data['total'] - data['fn_correct']
    n_incorrect = data['resp_correct'] - data['fn_correct']

    noex_only_corr = int(has_correct and noex_n_incorrect==0)
    only_corr = int(has_correct and n_incorrect==0)

    cdx_good_has_corr_arr[i,j] = has_correct
    cdx_noex_good_n_incorr_arr[i,j] = noex_n_incorrect
    cdx_good_n_incorr_arr[i,j] = n_incorrect

    cdx_noex_good_only_corr_arr[i,j] = noex_only_corr
    cdx_good_only_corr_arr[i,j] = only_corr
    c += 1

if __name__ == "__main__":
    print(c)
    print((~np.isnan(cdx_good_has_corr_arr)).astype(np.int32))
    print(np.sum((~np.isnan(cdx_good_has_corr_arr)).astype(np.int32)))

    print("Codex Has Correct:", np.nanmean(cdx_good_has_corr_arr))
    print("Codex Only Correct:", np.nanmean(cdx_good_only_corr_arr))
    print("Codex Avg Incorrect:", np.nanmean(cdx_good_n_incorr_arr))



    


