import pytest
def main():
    # Run pytest tests
    args = ["-m replay_buffer"]
    pytest.main() # TODO: pass args to here and make args accept command line argumets 


if __name__ == main:
    main()
