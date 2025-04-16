class RouteNode:
    def __init__(self):
        self.children = {}  # dict: segment -> RouteNode
        self.result = None  # value at end of full path


class MiddlewareRouter:
    def __init__(self):
        self.root = RouteNode()

    def add_route(self, path: str, result: str) -> None:
        """ Adds exact path with no wildcards """
        segments = path.strip("/").split("/")
        current = self.root
        for segment in segments:
            if segment not in current.children:
                current.children[segment] = RouteNode()
            current = current.children[segment]
        current.result = result

    def call_route(self, path: str) -> str:
        """ Matches path, allowing `*` in input path to match any segment """
        segments = path.strip("/").split("/")
        return self._dfs_match(self.root, segments, 0) or -1

    def _dfs_match(self, node: RouteNode, segments: list, index: int) -> str or None:
        if index == len(segments):
            return node.result

        segment = segments[index]
        results = []

        # If current segment is '*', try all children
        if segment == "*":
            for child in node.children.values():
                result = self._dfs_match(child, segments, index + 1)
                if result is not None:
                    results.append(result)
        else:
            # Exact match only
            if segment in node.children:
                result = self._dfs_match(node.children[segment], segments, index + 1)
                if result is not None:
                    results.append(result)

        return results[0] if results else None

if __name__ == "__main__":
    router = MiddlewareRouter()
    router.add_route("/home/about", "AboutPage")
    router.add_route("/home/contact", "ContactPage")
    router.add_route("/user/profile", "UserProfile")
    router.add_route("/user/settings", "UserSettings")

    print(router.call_route("/home/about"))     # "AboutPage"
    print(router.call_route("/home/*"))         # "AboutPage" or "ContactPage" (first match)
    print(router.call_route("/user/*"))         # "UserProfile" or "UserSettings"
    print(router.call_route("/admin/*"))        # -1
    print(router.call_route("/user/profile"))   # "UserProfile"
