from apprentice.agents.cre_agents.process_learning.process_learning import find_edits 


def test_find_edits():
    # 0. Aligned
    print("0.")
    seq1 = 'ABCDEF'
    find_edits(seq1, seq1)
    print()

    # 1. Unorder
    print("1.")
    seq2 = 'CBAFED'
    find_edits(seq1, seq2)
    print([('unorder', 0, 3), ('unorder', 3, 6)],'\n')


    # 2. Unorder + Aligned
    print("2.")
    seq2 = 'ACBDFE'
    find_edits(seq1, seq2)
    print([('unorder', 1, 3), ('unorder', 4, 6)],'\n')


    # # 3. Delete
    print("3.")
    seq2 = 'ACDF'
    find_edits(seq1, seq2)
    print([('delete', 1), ('delete', 4)],'\n')

    # # 4. Unorder + Delete
    print("4.")
    seq2 = 'CBED'
    find_edits(seq1, seq2)
    print([('delete', 0), ('unorder', 1, 3), ('unorder', 3, 5), ('delete', 5)],'\n')


    # # 5. Unorder Subsume Delete
    print("5.")
    seq2 = 'CAFD'
    find_edits(seq1, seq2)
    print([('unorder', 0, 3), ('delete', 1), ('unorder', 3, 6), ('delete', 4)],'\n')

    # # 6. Insert
    print("6.")
    seq2 = 'XABCDEFY'
    find_edits(seq1, seq2)
    print([('insert', 0),('insert', 6)],'\n')

    # # 7. Unorder + Insert
    print("7.")
    seq2 = 'XCBAFEDY'
    find_edits(seq1, seq2)
    print([('insert', 0), ('unorder', 0, 3), ('unorder', 3, 6), ('insert', 6)],'\n')

    # # 8. Unorder Subsume Insert
    print("8.")
    seq2 = 'CXBAFEYD'
    find_edits(seq1, seq2)
    print([('unorder', 0, 3), ('insert', 1), ('unorder', 3, 6), ('insert', 5)],'\n')

    # # 9. Replace
    print("9.1")
    seq2 = 'XYZDEF'
    find_edits(seq1, seq2)
    print([('replace', 0, 3)],'\n')

    print("9.2")
    seq2 = 'ABXYEF'
    find_edits(seq1, seq2)
    print([('replace', 2, 4)],'\n')

    print("9.3")
    seq2 = 'ABCXYZ'
    find_edits(seq1, seq2)
    print([('replace', 3, 6)],'\n')

    # # 10. Unorder + Replace
    print("10.")
    seq2 = 'BAXYFE'
    find_edits(seq1, seq2)
    print([('unorder', 0, 2), ('replace', 2, 4), ('unorder', 4, 6)],'\n')

    # # 11. Unorder + Replace
    print("11.")
    seq2 = 'CXAFYD'
    find_edits(seq1, seq2)
    print([('unorder', 0, 3), ('replace', 1, 2), ('unorder', 3, 6), ('replace', 4, 5)],'\n')


if(__name__ == "__main__"):
    test_find_edits()
