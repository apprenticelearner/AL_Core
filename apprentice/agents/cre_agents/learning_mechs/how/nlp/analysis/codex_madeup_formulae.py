stats = {}
def codex_eval(num, goal, _fn_correct=None): 
    
    data = stats.get(num, {**{
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
    print(f"{num} : ({int(has_correct)}/{n_incorrect+has_correct}/{n_bad_fn+has_correct}))")

# ---------------------------------------------------------
# 1: ((3-8) / Double(2)) + 1) == -0.25
# "Divide 3 minus 8 and twice 2, then increment."

@codex_eval(1, -0.25)
def foo():
    ''''Divide 3 minus 8 and twice 2, then increment.'''
    return 3 - 8 / 2 * 2 + 1

@codex_eval(1, -0.25)
def foo():
    ''''Divide 3 minus 8 and twice 2, then increment.'''
    return (3 - 8) / 2 * 2 + 1

@codex_eval(1, -0.25)
def foo():
    ''''Divide 3 minus 8 and twice 2, then increment.'''
    return (3 - 8) / (2 * 2) + 1

@codex_eval(1, -0.25, False)
def foo():
    ''''Divide 3 minus 8 and twice 2, then increment.'''
    return ((3 - 8) / 2) * 2 + 1

show_stats(1)


# ---------------------------------------------------------
# 1.5 : sdifjsodifjosidjfsd
# ""

@codex_eval(11, 13)
def foo():
    '''Decrement 3, then multiply it by 6, and add 1.'''
    return 3 * 6 - 1


@codex_eval(11, 13)
def foo():
    '''Decrement 3, then multiply it by 6, and add 1.'''
    return 3 * 6 + 1


@codex_eval(11, 13)
def foo():
    '''Decrement 3, then multiply it by 6, and add 1.'''
    return 3 - 1 * 6 + 1


@codex_eval(11, 13)
def foo():
    '''Decrement 3, then multiply it by 6, and add 1.'''
    a = 3
    a -= 1
    a *= 6
    a += 1
    return a


@codex_eval(11, 13)
def foo():
    '''Decrement 3, then multiply it by 6, and add 1.'''
    x = 3
    x -= 1
    x *= 6
    x += 1
    return x

show_stats(11)


# ---------------------------------------------------------
# 2: 7^(half(12) / 3) == 49.0
# "To get the exponent take half of 12 divided by 3. Then find 7 to the power of the exponent."

@codex_eval(2, 49, True)
def foo():
    ''''To get the exponent take half of 12 divided by 3. Then find 7 to the power of the exponent.'''
    return 7**(12/6)

@codex_eval(2, 49)
def foo():
    ''''To get the exponent take half of 12 divided by 3. Then find 7 to the power of the exponent.'''
    return (12/3)/2

@codex_eval(2, 49)
def foo():
    ''''To get the exponent take half of 12 divided by 3. Then find 7 to the power of the exponent.'''
    print(7 ** (12 / 3 / 2))

@codex_eval(2, 49)
def foo():
    ''''To get the exponent take half of 12 divided by 3. Then find 7 to the power of the exponent.'''
    return 7 ** (12 / 6 / 3)

@codex_eval(2, 49, True)
def foo():
    ''''To get the exponent take half of 12 divided by 3. Then find 7 to the power of the exponent.'''
    return 7 ** (12 / 3 / 2)

@codex_eval(2, 49, True)
def foo():
    ''''To get the exponent take half of 12 divided by 3. Then find 7 to the power of the exponent.'''
    x = 12 / 3 / 2
    return 7 ** x

show_stats(2)

# ---------------------------------------------------------
# 3: Square(Square(4)-6) == 100
# "Take the square of 4, then subtract 6 from it. Finally square the result")

@codex_eval(3, 100)
def foo():
    ''''Take the square of 4, then subtract 6 from it. Finally square the result.'''
    return 4**2 - 6**2

@codex_eval(3, 100, True)
def foo():
    ''''Take the square of 4, then subtract 6 from it. Finally square the result.'''
    return (4**2 - 6)**2

@codex_eval(3, 100)
def foo():
    ''''Take the square of 4, then subtract 6 from it. Finally square the result.'''
    return (4**2) - 6**2

@codex_eval(3, 100, True)
def foo():
    ''''Take the square of 4, then subtract 6 from it. Finally square the result.'''
    result = 4 ** 2
    result -= 6
    return result ** 2

@codex_eval(3, 100, True)
def foo():
    ''''Take the square of 4, then subtract 6 from it. Finally square the result.
    '''
    x = 4 ** 2
    y = x - 6
    return y ** 2

@codex_eval(3, 100, True)
def foo():
    ''''Take the square of 4, then subtract 6 from it. Finally square the result.

    Example
    -------
    >>> foo()
    4

    '''
    return (4**2 - 6)**2

show_stats(3)


# ---------------------------------------------------------
# 4: Ones(Double(11))+1) == 3
# "Take the one's digit of twice 11. Then increment it."

@codex_eval(4, 3, True)
def foo():
    '''Take the one's digit of twice 11. Then increment it.'''
    return (2*11) % 10 + 1

@codex_eval(4, 3, True)
def foo():
    '''Take the one's digit of twice 11. Then increment it.'''
    return (11 * 2) % 10 + 1

@codex_eval(4, 3)
def foo():
    '''Take the one's digit of twice 11. Then increment it.'''
    print((11 * 2) % 10 + 1)

@codex_eval(4, 3)
def foo():
    '''Take the one's digit of twice 11. Then increment it.'''
    x = 11
    y = 2 * x
    z = y % 10
    z += 1
    return z

@codex_eval(4, 3)
def foo():
    '''Take the one's digit of twice 11. Then increment it.'''
    x = 11
    x = x * 2
    x = x % 10
    x = x + 1
    return x

@codex_eval(4, 3)
def foo():
    '''Take the one's digit of twice 11. Then increment it.'''
    a = 11
    b = 11
    c = 2 * a + b
    d = c % 10
    e = d + 1
    return

show_stats(4)

# ---------------------------------------------------------
# 5: (5*2)**2+6 == 106
# "Find 5 times 2. Then square it and add 6 to it."

@codex_eval(5, 106)
def foo():
    '''Find 5 times 2. Then square it and add 6 to it.'''
    return (5 * 2) ** 2 + 6

@codex_eval(5, 106)
def foo():
    '''Find 5 times 2. Then square it and add 6 to it.'''
    return 5 * 2 ** 2 + 6

@codex_eval(5, 106)
def foo():
    '''Find 5 times 2. Then square it and add 6 to it.'''
    return 5 * 2 * 2 + 6

@codex_eval(5, 106)
def foo():
    '''Find 5 times 2. Then square it and add 6 to it.'''
    x = 5 * 2
    y = x ** 2
    z = y + 6

    return z

show_stats(5)

# ---------------------------------------------------------
# 6:  (double(8) + half(6))**2 == 361
# "Add double of 8 and half of 6. Then square the result."

@codex_eval(6, 361)
def foo():
    '''Add double of 8 and half of 6. Then square the result.'''
    return (8 + 6) ** 2

@codex_eval(6, 361, True)
def foo():
    '''Add double of 8 and half of 6. Then square the result.'''
    return (8*2 + 6/2)**2

@codex_eval(6, 361)
def foo():
    '''Add double of 8 and half of 6. Then square the result.'''
    return 2 * 8 + 6 / 2 ** 2

@codex_eval(6, 361, True)
def foo():
    '''Add double of 8 and half of 6. Then square the result.'''
    return ((8 * 2) + (6 / 2)) ** 2

show_stats(6)

# ---------------------------------------------------------
# 7: (9*3)/6-2 == 2.5
# "Divide 9 times 3 by 6, then subtract 2."

@codex_eval(7, 2.5, True)
def foo():
    '''Divide 9 times 3 by 6, then subtract 2.'''
    return (9 * 3 / 6) - 2

@codex_eval(7, 2.5)
def foo():
    '''Divide 9 times 3 by 6, then subtract 2.'''
    return (9 / 3) / 6 - 2

@codex_eval(7, 2.5, True)
def foo():
    '''Divide 9 times 3 by 6, then subtract 2.'''
    return 9 * 3 / 6 - 2

@codex_eval(7, 2.5, True)
def foo():
    '''Divide 9 times 3 by 6, then subtract 2.'''
    return ((9 * 3) / 6) - 2

show_stats(7)
# ---------------------------------------------------------
# 8: Double(13)+(5*7) == 61
# "Double 13 and add it to 5 times 7."

@codex_eval(8, 61, True)
def foo():
    '''Double 13 and add it to 5 times 7.'''
    return 13 * 2 + 5 * 7

@codex_eval(8, 61, True)
def foo():
    '''Double 13 and add it to 5 times 7.'''
    return 2 * 13 + 5 * 7

@codex_eval(8, 61, True)
def foo():
    '''Double 13 and add it to 5 times 7.'''
    return (13 * 2) + (5 * 7)

@codex_eval(8, 61, True)
def foo():
    '''Double 13 and add it to 5 times 7.'''
    return 5 * 7 + 13 * 2

@codex_eval(8, 61, True)
def foo():
    '''Double 13 and add it to 5 times 7.'''
    return (13 * 2) + (7 * 5)

show_stats(8)

# ---------------------------------------------------------
# 9: ((11-12)+(4**2))-6 == 12
# "Add 11 minus 9 and 4 squared. Then subtract 6"

@codex_eval(9, 12, True)
def foo():
    '''Add 11 minus 9 and 4 squared. Then subtract 6'''
    return 11-9 + 4**2 - 6

@codex_eval(9, 12)
def foo():
    '''Add 11 minus 9 and 4 squared. Then subtract 6'''
    return 11 + 9 + 4**2 - 6

@codex_eval(9, 12, True)
def foo():
    '''Add 11 minus 9 and 4 squared. Then subtract 6'''
    return 11 - 9 + (4 ** 2) - 6

show_stats(9)


# ---------------------------------------------------------
# 10: (Decrement(3)*6)+1 == 13
# "Decrement 3, then multiply it by 6, and add 1."

@codex_eval(10, 13)
def foo():
    '''Decrement 3, then multiply it by 6, and add 1.'''
    return 3 - 6 * 1

@codex_eval(10, 13)
def foo():
    '''Decrement 3, then multiply it by 6, and add 1.'''
    return 3 * 6 + 1

@codex_eval(10, 13, True)
def foo():
    '''Decrement 3, then multiply it by 6, and add 1.'''
    return 1 + 6 * (3 - 1)

@codex_eval(10, 13)
def foo():
    '''Decrement 3, then multiply it by 6, and add 1.'''
    return 3 - 1 * 6 + 1

@codex_eval(10, 13, True)
def foo():
    '''Decrement 3, then multiply it by 6, and add 1.'''
    return ((3 - 1) * 6) + 1

@codex_eval(10, 13)
def foo():
    '''Decrement 3, then multiply it by 6, and add 1.'''
    x = 3
    while x > 0:
        x -= 1
    return 6 * x + 1

@codex_eval(10, 13, True)
def foo():
    '''Decrement 3, then multiply it by 6, and add 1.'''
    x = 3
    x -= 1
    x *= 6
    x += 1
    return x


show_stats(10)
