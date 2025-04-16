import heapq

class FileCollectionAnalyzer:
    def __init__(self):
        self.collection_sizes = {}  # Only 1 map: collection -> total size
        self.total_size = 0         # Single scalar

    def add_file(self, file_name: str, file_size: int, collection_names: list):
        # We ignore file_name completely to save space
        for collection in collection_names:
            self.collection_sizes[collection] = self.collection_sizes.get(collection, 0) + file_size
        self.total_size += file_size

    def get_total_file_size(self):
        return self.total_size

    def get_top_k_collections_by_size(self, k: int):
        # O(C log K)
        return heapq.nlargest(k, self.collection_sizes.items(), key=lambda item: item[1])



import unittest

class TestFileCollectionAnalyzer(unittest.TestCase):
    def test_example_case(self):
        analyzer = FileCollectionAnalyzer()
        analyzer.add_file("file1.txt", 100, [])
        analyzer.add_file("file2.txt", 200, ["collection1"])
        analyzer.add_file("file3.txt", 200, ["collection1"34])
        analyzer.add_file("file4.txt", 300, ["collection2"])
        analyzer.add_file("file5.txt", 100, [])

        self.assertEqual(analyzer.get_total_file_size(), 900)
        top_2 = analyzer.get_top_k_collections_by_size(2)
        expected = [("collection1", 400), ("collection2", 300)]
        self.assertEqual(top_2, expected)

    def test_file_in_multiple_collections(self):
        analyzer = FileCollectionAnalyzer()
        analyzer.add_file("file1.txt", 100, ["c1", "c2"])
        analyzer.add_file("file2.txt", 200, ["c2"])
        self.assertEqual(analyzer.get_total_file_size(), 300)
        top_2 = analyzer.get_top_k_collections_by_size(2)
        expected = [("c2", 300), ("c1", 100)]
        self.assertEqual(top_2, expected)

    def test_empty_case(self):
        analyzer = FileCollectionAnalyzer()
        self.assertEqual(analyzer.get_total_file_size(), 0)
        self.assertEqual(analyzer.get_top_k_collections_by_size(1), [])

if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)

