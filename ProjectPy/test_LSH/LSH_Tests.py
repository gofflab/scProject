import unittest
import ProjectPy
from ProjectPy.LSH_Object import Hashing


def main():
    hasher = Hashing(5, 571, "retina_patterns.csv", "retina_patterns.csv")
    hasher2 = Hashing(5, 571, "retina_patterns.csv", "retina_patterns.csv", "retina_patterns.csv")
    print("fin constructor")
    list_clusters = hasher.hash_multiple_times_random(3)
    list_clusters = hasher2.hash_multiple_times_Chosen(3)
    print("done random")


if __name__ == '__main__':
    main()
    print("woohoo")

