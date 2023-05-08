import pytest
def main():
    # Run pytest tests
    args = ["-m replay_buffer or forklift_env"]
    pytest.main(args) # TODO: pass args to here and make args accept command line argumets 


if __name__ == main:
    main()
