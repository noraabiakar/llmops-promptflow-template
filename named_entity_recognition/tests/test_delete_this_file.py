def test_print():
    try:
        print("Hello") is None
    except:
        print("OH NO! Test print function failed.")
        assert False
