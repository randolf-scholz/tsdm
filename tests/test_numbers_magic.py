from tsdm.numbers_magic import addition


def test_addition_function():
    # given
    summand_one = 1
    summand_two = 2

    # when
    result = addition(summand_one, summand_two)

    # then
    assert result == 3
