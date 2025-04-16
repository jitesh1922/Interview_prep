class FileSystemNode:
    def __init__(self):
        self.children = {}  # maps segment -> FileSystemNode
        self.value = None   # value at this path, if any

#https://leetcode.com/problems/design-file-system/
class FileSystem:
    def __init__(self):
        self.root = FileSystemNode()

    def createPath(self, path: str, value: int) -> bool:
        try:
            if not path or path == "/" or path[-1] == "/":
                return False

            parts = path.strip("/").split("/")
            current = self.root

            for i in range(len(parts)):
                part = parts[i]
                if i == len(parts) - 1:  # last segment
                    if part in current.children:
                        return False  # path already exists
                    current.children[part] = FileSystemNode()
                    current.children[part].value = value
                    return True
                if part not in current.children:
                    return False  # parent path missing
                current = current.children[part]

            return False
        except Exception as e:
            print(f"Error in createPath: {e}")
            return False

    def get(self, path: str) -> int:
        try:
            if not path or path == "/" or path[-1] == "/":
                return -1

            parts = path.strip("/").split("/")
            current = self.root

            for part in parts:
                if part not in current.children:
                    return -1
                current = current.children[part]

            return current.value if current.value is not None else -1
        except Exception as e:
            print(f"Error in get: {e}")
            return -1


import unittest

class TestFileSystem(unittest.TestCase):
    def test_basic_operations(self):
        fs = FileSystem()
        self.assertTrue(fs.createPath("/a", 1))
        self.assertFalse(fs.createPath("/a", 2))      # already exists
        self.assertTrue(fs.createPath("/a/b", 2))
        self.assertFalse(fs.createPath("/c/d", 3))    # parent /c missing
        self.assertEqual(fs.get("/a"), 1)
        self.assertEqual(fs.get("/a/b"), 2)
        self.assertEqual(fs.get("/c"), -1)
        self.assertEqual(fs.get("/"), -1)             # root is not allowed

if __name__ == "__main__":
    #unittest.main(argv=[''], exit=False)

    fs = FileSystem()
    print(fs.createPath("/a", 1))
    print(fs.createPath("/a", 2))
    print(fs.createPath("/a/b", 2))
    print(fs.createPath("/c/d", 3))
    print(fs.get("/a"))
    print(fs.get("/a/b"))
    print(fs.get("/c"))
    print(fs.get("/"))

