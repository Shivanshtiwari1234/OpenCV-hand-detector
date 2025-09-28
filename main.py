from multiprocessing import set_start_method
import handtrace

if __name__ == "__main__":
    set_start_method("spawn")
    handtrace.main()
