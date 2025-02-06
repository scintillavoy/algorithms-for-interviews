# Algorithms for interviews

- This is a public version of the document I personally organized to prepare for data structures and algorithms interviews.
- This document serves as a cheat sheet, and many detailed explanations have been omitted.
- All example code is written in C++.
- Some of the content here may not be suitable for interviews (e.g. 2-SAT).
- Some of the links lead to articles in Korean.

## Data structures

### Array

- `size()` returns unsigned and it should not be compared with negatives.
- `vector<bool>` uses a proxy, which is a prvalue, for the efficiency so use rvalue reference (`auto &&`) in range for loop.
  - <https://stackoverflow.com/questions/17794569/why-isnt-vectorbool-a-stl-container>
  - <https://stackoverflow.com/questions/34079390/range-for-loops-and-stdvectorbool>

```cpp
#include <vector>
vector<int> vec(N, 0);
vec[0] == 0;
vector<vector<int>> vec2(N, vector<int>(M, 0));
vec.insert(pos, 0); // O(vec.end() - pos)

// Initialize with elements.
vector<int> vec3 = {1, 2, 3, 4};
```

- (Korean) <https://namwhis.tistory.com/entry/C-%EB%B0%B0%EC%97%B4-%EC%B4%88%EA%B8%B0%ED%99%94-stdfill-stdfilln-%EC%A0%95%EB%A6%AC>

```cpp
// Elements are initialized to 0 if it is declared as static or is global.
int arr[1][2][3];
int arr[1][2][3] = {0}; // Set elements to 0.

#include <algorithm>
fill_n(&arr[0][0][0], sizeof(arr) / sizeof(arr[0][0][0]), value); // Set elements to value.
```

### String

```cpp
#include <string>
string str("abcde");
string str(vec.begin(), vec.end()); // Construct a string from vector.
str.substr(pos = 0, count = npos);  // [pos, pos + count)
str.substr(1, 3) == "bcd";          // O(n)
str.find("c", 0) == 2;              // O(n + m) (depends on the implementation)
str.find("f") == string::npos;
to_string(val);                     // Convert an arithmetic type to a string.
// To convert a string to an arithmetic type, use the following functions:
//   stoi, stol, stoul, stoll, stoull, stof, stod, stold
```

```cpp
#include <cctype>
int tolower(int ch);
int toupper(int ch);
int isupper(int ch);
int islower(int ch);
int isdigit(int ch); // If `ch` is a decimal, return some value other than 0.
int isalpha(int ch);
```

```cpp
#include <iostream>
#include <limits>
#include <string>
cin.ignore(numeric_limits<streamsize>::max(), '\n'); // Consume newline character before using `getline`.
getline(cin, str); // Read from `cin` until newline and save to `str` except newline.
```

```cpp
#include <sstream>
#include <string>
#include <vector>
string s("AB CD EFG HI");

// istringstream
istringstream iss(s);        // Copy `s` to `iss`. The same as `istringstream iss; iss.str(s);`.
vector<string> words;
string word;
while (getline(iss, word, delimiter)) { // Split by delimiter.
  words.push_back(word);
}
while (iss >> word) {        // Split by whitespace characters (e.g. space, newline, ...)
  words.push_back(word);     // {"AB", "CD", "EFG", "HI"}
}
iss.clear();                 // If you want to reuse it, call clear() and
iss.str(s);                  // Copy a string.

// ostringstream
ostringstream oss;
string line;
while (getline(cin, line)) { // If the inputs are "ABC\n", "DEF\n", "GHI\n",
  oss << line;
}
oss.str();                   // == "ABCDEFGHI"
ostringstream oss2(s);       // Copy `s` to `oss2`. The same as `ostringstream oss2; oss2.str(s);`.
```

#### Knuth-Morris-Pratt (KMP) algorithm

- <https://www.geeksforgeeks.org/kmp-algorithm-for-pattern-searching/>
- <https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/description/>

```cpp
// Time complexity: O(n + m)
//   n: the length of the text.
//   m: the length of the pattern.
string text = "AABAACAADAABAABA";
//             ^        ^  ^
string pattern = "AABA";
// result: [0, 9, 12]

// Preprocess the pattern.
vector<int> kmp_table(pattern.size());
for (int i = 1, j = 0; i < pattern.size(); ++i) {
  while (j > 0 && pattern[i] != pattern[j]) {
    j = kmp_table[j - 1];
  }
  if (pattern[i] == pattern[j]) {
    ++j;
  }
  kmp_table[i] = j;
}

// Find occurrences of the pattern in the text.
vector<int> result;
for (int i = 0, j = 0; i < text.size(); ++i) {
  while (j > 0 && text[i] != pattern[j]) {
    j = kmp_table[j - 1];
  }
  if (text[i] == pattern[j]) {
    ++j;
    if (j == pattern.size()) {
      result.push_back(i + 1 - j);
      j = kmp_table[j - 1];
    }
  }
}
```

### Stack

```cpp
#include <stack>
stack<T> stk;
stk.empty() == true;            // O(1)
stk.push(v1);                   // O(1) for deque
stk.push(v2);
stk.size() == 2;                // O(1)
stk.top() == v2;                // O(1)
stk.pop();                      // O(1) for deque
stk.top() == v1;
```

### Queue

```cpp
#include <queue>
queue<T> q;
q.empty() == true;              // O(1)
q.push(v1);                     // O(1) for deque
q.push(v2);
q.size() == 2;                  // O(1)
q.front() == v1;                // O(1)
q.back() == v2;                 // O(1)
q.pop();                        // O(1) for deque
q.front() == v2;
```

```cpp
#include <deque>
deque<int> dq;
dq.front();
dq.back();
dq.push_front(1);
dq[0] == 1;
dq.push_back(2);
dq.pop_front();
dq.pop_back();
```

### Priority queue

```cpp
#include <queue>
priority_queue<int> pq;         // Max heap
priority_queue<int, vector<int>, greater<int>> pq2;  // Min heap

// Custom comparator
auto comp = [](const int a, const int b){ return a < b; };
priority_queue<int, vector<int>, decltype(comp)> pq3(comp); // Until C++17.
priority_queue<int, vector<int>, decltype(comp)> pq3; // Since C++20.
// `comp` has to be passed as an argument (like C++17) if it captures.

pq.empty() == true;             // O(1)
pq.push(3);                     // O(logn)
pq.push(5);
pq.size() == 2;                 // O(1)
pq.top() == 5;                  // O(1)
pq.pop();                       // O(logn)
pq.top() == 3;
```

### Linked list

```cpp
#include <forward_list>
// An iterator is invalidated only when the corresponding element is deleted.
forward_list<T> fl;             // Singly-linked list
fl.front();                     // O(1)
fl.before_begin();              // O(1)
fl.begin();                     // O(1), is equal to end() if the list is empty.
fl.end();                       // O(1)
fl.empty() == true;             // O(1)
fl.insert_after(it, v);         // O(1), returns iterator to the inserted element.
fl.erase_after(it);             // O(1), returns iterator to the element following the erased one.
fl.push_front(v);               // O(1)
fl.pop_front();                 // O(1)
fl.merge(l2);                   // O(n), merges two sorted lists into one.
fl.splice_after(it, fl2);       // O(n), moves elements from another forward_list to *this.
fl.splice_after(it, fl2, it2);  // O(1)
fl.splice_after(it, fl2, first, last);  // O(n)
fl.sort();                      // O(nlogn)
```

```cpp
// Find the (right) middle of the list.
ListNode *slow = head;
ListNode *fast = head;
while (fast != nullptr && fast->next != nullptr) {
  // To find the left middle,
  // while (fast->next != nullptr && fast->next->next != nullptr) {
  slow=slow->next;
  fast=fast->next->next;
}
```

```cpp
#include <list>
// An iterator is invalidated only when the corresponding element is deleted.
list<T> l;                      // Doubly-linked list
l.front();                      // O(1)
l.back();                       // O(1)
l.begin();                      // O(1), is equal to end() if the list is empty.
l.end();                        // O(1)
l.empty();                      // O(1)
l.size();                       // O(1)
l.insert(it, v);                // O(1), returns iterator pointing to the inserted value.
l.erase(it);                    // O(1), returns iterator following the last removed element.
l.push_back(v);                 // O(1)
l.pop_back();                   // O(1)
l.push_front(v);                // O(1)
l.pop_front();                  // O(1)
l.merge(l2);                    // O(n), merges two sorted lists into one.
l.splice(it, l2);               // O(1), transfers elements from one list to another.
l.splice(it, l2, it2);          // O(1)
l.splice(it, l2, first, last);  // O(1) if two lists are the same object, O(n) otherwise.
l.sort();                       // O(nlogn)
```

### Hash table

```cpp
#include <unordered_set>
unordered_set<K> us;

// Use a tuple as a key.
// Refer to https://stackoverflow.com/questions/2620862/using-custom-stdset-comparator.
// Caution: XOR is not a good choice for the hash function.
auto hash = [](const tuple<int, int, int> &x) {
  return get<0>(x) ^ get<1>(x) ^ get<2>(x);
};
unordered_set<tuple<int, int, int>, decltype(hash)> us(0, hash); // Until C++17.
unordered_set<tuple<int, int, int>, decltype(hash)> us; // Since C++20.
// `hash` has to be passed as an argument (like C++17) if it captures.

us.count(v);                    // Average: O(1), worst: O(n)
us.contains(v);                 // Average: O(1), worst: O(n). Since C++20.
us.insert(v);                   // Average: O(1), worst: O(n)
us.erase(v);                    // Average: O(1), worst: O(n)
us.erase(it);                   // Average: O(1), worst: O(n)
us.size();                      // O(1)
us.begin();                     // O(1)
us.end();                       // O(1)

// unordered_multiset
unordered_multiset<K> ums;

ums.erase(v);   // O(count)
ums.extract(v); // O(1). Since C++17.
```

```cpp
#include <unordered_map>
unordered_map<K, V> um;

// Use a tuple as a key.
auto hash = [](const tuple<int, int, int> &x) {
  return get<0>(x) ^ get<1>(x) ^ get<2>(x);
};
unordered_map<tuple<int, int, int>, bool, decltype(hash)> um(0, hash); // Until C++17.
unordered_map<tuple<int, int, int>, bool, decltype(hash)> um; // Since C++20.
// `hash` has to be passed as an argument (like C++17) if it captures.

um.count(v);                    // Average: O(1), worst: O(n)
um.contains(v);                 // Average: O(1), worst: O(n). Since C++20.
um.insert(v);                   // Average: O(1), worst: O(n)
um.erase(v);                    // Average: O(1), worst: O(n)
um.erase(it);                   // Average: O(1), worst: O(n)
um.size();                      // O(1)
um.begin();                     // O(1)
um.end();                       // O(1)

// Caution: This inserts an element if it does not exist (it will be value-initialized).
um[v];

// unordered_multimap
unordered_multimap<K, V> umm;

umm.erase(v);   // O(count)
umm.extract(v); // O(1). Since C++17.
```

### Binary search tree (BST)

```cpp
#include <set>
set<K> s;                       // Usually implemented as a red-black tree.
set<K, greater<K>> s2;

// Custom comparator
auto comp = [](const int a, const int b){ return a < b; };
set<K, decltype(comp)> s3(comp); // Until C++17.
set<K, decltype(comp)> s3; // Since C++20.
// `comp` has to be passed as an argument (like C++17) if it captures.

s.count(v);                     // O(logn)
s.contains(v);                  // O(logn). Since C++20.
s.insert(v);                    // O(logn)
s.erase(v);                     // O(logn)
s.erase(it);                    // Amortized: O(1)
s.begin();                      // O(1)
s.end();                        // O(1)

// multiset
multiset<K> ms;
multiset<K, greater<K>> ms2;

ms.erase(v);   // O(logn + count). It erases all elements with the key.
ms.extract(v); // O(logn). Since C++17. It erases the first element with the key.
```

```cpp
#include <map>
map<K, V> m;                    // Usually implemented as a red-black tree.
map<K, V, greater<K>> m2;

// Custom comparator
auto comp = [](const int a, const int b){ return a < b; };
map<K, V, decltype(comp)> m3(comp); // Until C++17.
map<K, V, decltype(comp)> m3; // Since C++20.
// `comp` has to be passed as an argument (like C++17) if it captures.

m.count(v);                     // O(logn)
m.contains(v);                  // O(logn). Since C++20.
m.insert(v);                    // O(logn)
m.erase(v);                     // O(logn)
m.erase(it);                    // Amortized: O(1)
m.begin();                      // O(1)
m.end();                        // O(1)

// Caution: This inserts an element if it does not exist (it will be value-initialized).
m[v];

// multimap
multimap<K, V> mm;
multimap<K, V, greater<K>> mm2;

mm.erase(v);   // O(logn + count)
mm.extract(v); // O(logn). Since C++17.
```

### Tree

- A graph is a tree if and only if:
  - It is fully connected.
  - It has no cycles.
- Traversal
  - Pre-order + null node: are structures of the trees the same?
    ```cpp
    void preorder(int curr) {
      if (curr == -1) return;
      cout << curr;
      preorder(tree[curr].left);
      preorder(tree[curr].right);
    }
    ```
  - In-order: are elements in sorted order?
    ```cpp
    void inorder(int curr) {
      if (curr == -1) return;
      inorder(tree[curr].left);
      cout << curr;
      inorder(tree[curr].right);
    }
    ```
  - Post-order
    ```cpp
    void postorder(int curr) {
      if (curr == -1) return;
      postorder(tree[curr].left);
      postorder(tree[curr].right);
      cout << curr;
    }
    ```
- Morris traversal (traversal with space complexity of O(1))
  - <https://leetcode.com/problems/sum-root-to-leaf-numbers/editorial/>
  - <https://www.geeksforgeeks.org/inorder-tree-traversal-without-recursion-and-without-stack/>
- Lowest common ancestor
  - (Korean) <https://jason9319.tistory.com/90>

### Disjoint set (Union-Find)

- <https://www.geeksforgeeks.org/disjoint-set-data-structures/>
- <https://www.boost.org/doc/libs/1_64_0/libs/disjoint_sets/disjoint_sets.html>

```cpp
vector<int> parents(N);
for (int i = 0; i < N; ++i) {
  parents[i] = i;
}
vector<int> ranks(N);

// Time complexity: O(a(N)) (nearly constant)
int find_set(int x) {
  if (parents[x] != x) {
    parents[x] = find_set(parents[x]); // Path compression
  }
  return parents[x];
}

// Time complexity: O(a(N)) (nearly constant)
void union_set(int x, int y) {
  x = find_set(x);
  y = find_set(y);
  if (x == y) {
    return;
  }
  if (ranks[x] > ranks[y]) { // Union by rank
    swap(x, y);
  } else if (ranks[x] == ranks[y]) {
    ++ranks[y];
  }
  parents[x] = y;
}
```

### Segment tree

- (Korean) <https://book.acmicpc.net/ds/segment-tree>

```cpp
vector<int> A(N); // The index starts from 0.
vector<int> tree(1 << ((int)ceil(log2(N)) + 1)); // N * 4

// Time complexity: O(N)
void init(int node, int start, int end) {
  if (start == end) {
    tree[node] = A[start];
    return;
  }
  int mid = (start + end) / 2;
  init(node * 2, start, mid);
  init(node * 2 + 1, mid + 1, end);
  tree[node] = tree[node * 2] + tree[node * 2 + 1];
}
init(1, 0, N - 1);

// Time complexity: O(logN)
void update(int node, int start, int end, int index, int value) {
  if (index < start || end < index) {
    return;
  }
  if (start == end) {
    A[index] = value;
    tree[node] = A[index];
    return;
  }
  int mid = (start + end) / 2;
  update(node * 2, start, mid, index, value);
  update(node * 2 + 1, mid + 1, end, index, value);
  tree[node] = tree[node * 2] + tree[node * 2 + 1];
}
update(1, 0, N - 1, index, value);

// Time complexity: O(logN)
int query(int node, int start, int end, int left, int right) {
  if (left > end || right < start) {
    return 0;
  }
  if (left <= start && end <= right) {
    return tree[node];
  }
  int mid = (start + end) / 2;
  int left_sum = query(node * 2, start, mid, left, right);
  int right_sum = query(node * 2 + 1, mid + 1, end, left, right);
  return left_sum + right_sum;
}
query(1, 0, N - 1, left, right);

// For the index starting from 1,
// vector<int> A(N + 1);
// init(1, 1, N);
// query(1, 1, N, left, right);
// update(1, 1, N, index, value);
```

### Trie

- (Korean) <https://ansohxxn.github.io/algorithm/trie/>

```cpp
#include <string>
#include <unordered_map>
using namespace std;

class Trie {
 public:
  Trie() { root = new TrieNode(); }
  ~Trie() { delete_trie(root); }

  void insert(const string& word) {
    TrieNode *curr = root;
    for (const auto c : word) {
      if (!curr->children.contains(c)) {
        curr->children[c] = new TrieNode();
      }
      curr = curr->children[c];
    }
    curr->is_end = true;
  }

  bool search(const string& word) {
    TrieNode *curr = root;
    for (const auto c : word) {
      if (!curr->children.contains(c)) {
        return false;
      }
      curr = curr->children[c];
    }
    return curr->is_end;
  }

 private:
  struct TrieNode {
    bool is_end = false;
    unordered_map<char, TrieNode *> children;
  };

  TrieNode *root;

  void delete_trie(TrieNode *node) {
    if (node == nullptr) {
      return;
    }
    for (auto it = node->children.begin(); it != node->children.end(); ++it) {
      delete_trie(it->second);
    }
    delete node;
  }
};
```

## Graph algorithms

### Breadth-first search (BFS) / 0-1 BFS

```cpp
// Breadth-first search
// Time complexity: O(V + E)
vector<vector<int>> graph(N);
for (int i = 0; i < M; ++i) {
  int from, to;
  cin >> from >> to;
  graph[from].push_back(to);
}
vector<int> parent(N, -1);
vector<bool> visited(N);
vector<int> distances(N, numeric_limits<int>::max());
vector<int> start_time(N);
int current_time = 0;
queue<int> q;

visited[source] = true;
distances[source] = 0;
start_time[source] = ++current_time;
q.push(source);
while (!q.empty()) {
  int curr = q.front();
  q.pop();
  for (const auto adj : graph[curr]) {
    if (visited[adj]) {
      continue;
    }
    visited[adj] = true;
    parent[adj] = curr;
    distances[adj] = distances[curr] + 1;
    start_time[adj] = ++current_time;
    q.push(adj);
  }
}
```

```cpp
// 0-1 breadth-first search
// Time complexity: O(V + E)
// Weights should be 0 or 1.
vector<vector<pair<int, int>>> graph(N);
for (int i = 0; i < M; ++i) {
  int from, to, weight;
  cin >> from >> to >> weight;
  graph[from].push_back({weight, to});
}
vector<int> parent(N, -1);
vector<int> distances(N, numeric_limits<int>::max());
deque<int> dq;

distances[source] = 0;
dq.push_front(source);
while (!dq.empty()) {
  int curr = dq.front();
  dq.pop_front();
  for (const auto [weight, adj] : graph[curr]) {
    if (distances[adj] > distances[curr] + weight) {
      distances[adj] = distances[curr] + weight;
      parent[adj] = curr;
      if (weight == 0) {
        dq.push_front(adj);
      } else {
        dq.push_back(adj);
      }
    }
  }
}
```

### Depth-first search (DFS)

```cpp
// Depth-first search
// Time complexity: O(V + E)
vector<vector<int>> graph(N);
for (int i = 0; i < M; ++i) {
  int from, to;
  cin >> from >> to;
  graph[from].push_back(to);
}
vector<bool> visited(N);
vector<int> parent(N, -1);
vector<int> start_time(N);
vector<int> finish_time(N);
int current_time = 0;

void dfs(int curr) {
  visited[curr] = true;
  start_time[curr] = ++current_time;
  for (const auto adj : graph[curr]) {
    if (visited[adj]) {
      continue;
    }
    parent[adj] = curr;
    dfs(adj);
  }
  finish_time[curr] = ++current_time;
}
dfs(source);
```

### Topological sort

```cpp
// Kahn's algorithm
// Time complexity: O(V + E)
// The graph should be a direct acyclic graph.
vector<vector<int>> graph(N);
vector<int> in_degree(N);
for (int i = 0; i < M; ++i) {
  int from, to;
  cin >> from >> to;
  graph[from].push_back(to);
  ++in_degree[to];
}

queue<int> q;
for (int i = 0; i < N; ++i) {
  if (in_degree[i] == 0) {
    q.push(i);
  }
}

vector<int> finished;
while (!q.empty()) {
  int curr = q.front();
  q.pop();
  finished.push_back(curr);
  for (const auto &adj : graph[curr]) {
    --in_degree[adj];
    if (in_degree[adj] == 0) {
      q.push(adj);
    }
  }
}

if (finished.size() != N) {
  // There is a cycle in the graph.
}
```

```cpp
// Using DFS
// Time complexity: O(V + E)
// The graph should be a direct acyclic graph.
vector<vector<int>> graph(N);
for (int i = 0; i < M; ++i) {
  int from, to;
  cin >> from >> to;
  graph[from].push_back(to);
}
vector<bool> visited(N);
vector<int> finished; // In the reversed order.

void dfs(int curr) {
  visited[curr] = true;
  for (const auto &adj : graph[curr]) {
    if (visited[adj]) continue;
    dfs(adj);
  }
  finished.push_back(curr);
}

for (int i = 0; i < N; ++i) {
  if (visited[i]) continue;
  dfs(i);
}
```

### Strongly connected components (SCC)

- A graph composed of SCCs is a DAG.
- Tarjan's algorithm uses lowlinks instead of lowpoints.
  - Lowpoint: the smallest index reachable from some vertex through any part of the graph.
  - Lowlink: the smallest index reachable from some vertex through the DFS subtree of the vertex.
  - SCC are correctly constructed by using lowpoints, but lowpoints are not calculated correctly by their definition.
- <https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm>
- <https://www.baeldung.com/cs/scc-tarjans-algorithm>
- (Korean) <https://com24everyday.tistory.com/149>

```cpp
// Tarjan's algorithm
// Time complexity: O(V + E)
vector<vector<int>> graph(N);
for (int i = 0; i < M; ++i) {
  int from, to;
  cin >> from >> to;
  graph[from].push_back(to);
}
vector<int> scc_id(N);
int current_scc_id = 0;
vector<int> start_time(N);
int current_time = 0;
vector<int> lowlink(N); // A return value of dfs can be used instead of this vector.
stack<int> s;

void dfs(int curr) {
  start_time[curr] = ++current_time;
  lowlink[curr] = start_time[curr];
  s.push(curr);
  for (const auto &adj : graph[curr]) {
    if (start_time[adj] == 0) {
      dfs(adj);
      lowlink[curr] = min(lowlink[curr], lowlink[adj]);
    } else if (scc_id[adj] == 0) {
      lowlink[curr] = min(lowlink[curr], start_time[adj]);
    }
  }
  if (lowlink[curr] == start_time[curr]) {
    ++current_scc_id;
    while (true) {
      int t = s.top();
      s.pop();
      scc_id[t] = current_scc_id;
      if (t == curr) {
        break;
      }
    }
  }
}

for (int i = 0; i < N; ++i) {
  if (start_time[i] == 0) {
    dfs(i);
  }
}
```

- (Korean) <https://gaussian37.github.io/math-algorithm-scc/>

```cpp
// Kosaraju's algorithm
// Time complexity: O(V + E)
vector<vector<int>> graph(N);
vector<vector<int>> graph_reversed(N);
for (int i = 0; i < M; ++i) {
  int from, to;
  cin >> from >> to;
  graph[from].push_back(to);
  graph_reversed[from].push_back(to);
}
vector<bool> visited(N);
stack<int> finished;
vector<int> scc_id(N);
int current_scc_id = 0;

void dfs(int curr) {
  visited[curr] = true;
  for (const auto &adj : graph[curr]) {
    if (visited[adj]) continue;
    dfs(adj);
  }
  finished.push(curr);
}

void dfs_reversed(int curr) {
  visited[curr] = true;
  scc_id[curr] = current_scc_id;
  for (const auto &adj : graph_reversed[curr]) {
    if (visited[adj]) continue;
    dfs_reversed(adj);
  }
}

for (int i = 0; i < N; ++i) {
  if (visited[i]) continue;
  dfs(i);
}

visited = vector<bool>(N);
while (!finished.empty()) {
  int curr = finished.top();
  finished.pop();
  if (scc_id[curr] != 0) continue;
  ++current_scc_id;
  dfs_reversed(curr);
}
```

### 2-SAT

- (Korean) <https://blog.naver.com/PostView.nhn?blogId=kks227&logNo=220803009418>

```cpp
// Time complexity: O(V + E)
inline int index_of(int x) { return x > 0 ? x * 2 - 1 : -x * 2 - 2; }

inline int not_of(int index) { return index ^ 1; }

vector<vector<int>> graph(2 * N);
for (int i = 0; i < M; ++i) {
  int a, b;
  cin >> a >> b;
  a = index_of(a);
  b = index_of(b);
  graph[not_of(a)].push_back(b);
  graph[not_of(b)].push_back(a);
}
vector<int> scc_id(2 * N);
int current_scc_id = 0;
vector<int> start_time(2 * N);
int current_time = 0;
vector<int> lowlink(2 * N);
stack<int> s;

void dfs(int curr) {
  // Find SCCs.
}

for (int i = 0; i < 2 * N; ++i) {
  if (start_time[i] == 0) {
    dfs(i);
  }
}

// Check if a solution exists.
for (int i = 1; i <= N; ++i) {
  if (scc_id[index_of(i)] == scc_id[index_of(-i)]) {
    cout << 0;
    return 0;
  }
}
cout << 1 << '\n';

// Find a solution.
for (int i = 1; i <= N; ++i) {
  cout << (scc_id[index_of(i)] < scc_id[index_of(-i)] ? 1 : 0) << ' ';
}
```

- Clause
  - $(\neg x_1 \lor x_2)$
- Conjunctive normal form (CNF)
  - $(\neg x_1 \lor x_2)\land(\neg x_2 \lor x_3)\land(x_1 \lor x_3)\land(x_3 \lor x_2)$
- 2-SAT problem
  - The maximum number of literals per clause is two.
- Example
  - $f=(x_1 \lor \neg x_2)\land(x_2 \lor \neg x_3)\land(x_3 \lor \neg x_1)\land(\neg x_4 \lor \neg x_2)$
  - For $f$ to be true, the following propositions must be true.
  - $(x_1 \lor \neg x_2)$
    - $\neg x_1 \to \neg x_2$
    - $x_2 \to x_1$
  - $(x_2 \lor \neg x_3)$
    - $\neg x_2 \to \neg x_3$
    - $x_3 \to x_2$
  - $(x_3 \lor \neg x_1)$
    - $\lnot x_3 \to \neg x_1$
    - $x_1 \to x_3$
  - $(\neg x_4 \lor \neg x_2)$
    - $x_4 \to \neg x_2$
    - $x_2 \to \neg x_4$
  - Build a graph with $x_n$ is a vertex and $\to$ is a edge.
  - Find SCCs.
  - For $f$ to be true, one SCC must not contain $x_n$ and $\lnot x_n$ at the same time.
  - To determine $x_n$ which makes $f$ true, mark $x_n$ as false if SCC id of $x_n$ is greater than the one of $\neg x_n$, mark as true otherwise.
  - The solution can also be obtained by topological sort.
    - Topologically sort SCCs (by descending order of SCC id).
    - When visiting $x_n$ which is not marked, mark it as false and $\lnot x_n$ as true.
- XOR: $x_1 \oplus x_2 = (x_1 \lor x_2) \land (\lnot x_1 \lor \lnot x_2)$
- Always true: $(x_n \lor x_n)$

### Minimum spanning tree (MST)

- (Korean) <https://godls036.tistory.com/26>

```cpp
// Kruskal's algorithm
// Time complexity: O(ElogV)
vector<tuple<int, int, int>> edges;
for (int i = 0; i < M; ++i) {
  int weight, from, to;
  cin >> weight >> from >> to;
  edges.push_back({weight, from, to});
}
sort(edges.begin(), edges.end());
vector<tuple<int, int, int>> mst_edges;

for (const auto &edge : edges) {
  auto [weight, from, to] = edge;
  if (find_set(from) == find_set(to)) {
    continue;
  }
  union_set(from, to);
  mst_edges.push_back(edge);
}
```

```cpp
// Prim's algorithm
// Time complexity: O(ElogV)
vector<vector<pair<int, int>>> graph(N);
for (int i = 0; i < M; ++i) {
  int weight, from, to;
  cin >> weight >> from >> to;
  graph[from].push_back({weight, to});
  graph[to].push_back({weight, from});
}
vector<bool> visited(N);
priority_queue<pair<int, int>, vector<pair<int, int>>,
               greater<pair<int, int>>>
    pq;
int mst_cost = 0;

pq.push({0, 0});
while (!pq.empty()) {
  auto [curr_weight, curr] = pq.top();
  pq.pop();
  if (visited[curr]) {
    continue;
  }
  visited[curr] = true;
  mst_cost += curr_weight;
  for (const auto [adj_weight, adj] : graph[curr]) {
    if (visited[adj]) {
      continue;
    }
    pq.push({adj_weight, adj});
  }
}
```

### Single-source shortest paths

#### Bellman-Ford

```cpp
// Bellman-Ford algorithm
// Time complexity: O(VE)
// Negative-weight cycles are detected.
// Negative-weight edges are ok.
vector<vector<pair<int, int>>> graph(N);
for (int i = 0; i < M; ++i) {
  int from, to, weight;
  cin >> from >> to >> weight;
  graph[from].push_back({weight, to});
}
vector<int> distances(N, numeric_limits<int>::max());
vector<int> parents(N, -1);
vector<int> cycles;

distances[source] = 0;
for (int i = 0; i < N; ++i) {
  for (int from = 0; from < N; ++from) {
    if (distances[from] == numeric_limits<int>::max()) {
      continue;
    }
    for (const auto [weight, to] : graph[from]) {
      if (distances[to] > distances[from] + weight) {
        if (i == N - 1) {
          cycles.push_back(from);
        }
        distances[to] = distances[from] + weight;
        parents[to] = from;
      }
    }
  }
}

if (!cycles.empty()) {
  // There is a negative-weight cycle in the graph.
}
```

#### Dijkstra's algorithm

- <https://stackoverflow.com/questions/6799172/negative-weights-using-dijkstras-algorithm/6799344#6799344>

```cpp
// Dijkstra's algorithm
// Time complexity: O(ElogV)
// No negative-weight cycles.
// No negative-weight edges.
// This implementation works even if there is a negative-weight edge,
// but the time complexity would not be O(ElogV).
vector<vector<pair<int, int>>> graph(N);
for (int i = 0; i < M; ++i) {
  int from, to, weight;
  cin >> from >> to >> weight;
  graph[from].push_back({weight, to});
}
vector<int> distances(N, numeric_limits<int>::max());
vector<int> parent(N, -1);
priority_queue<pair<int, int>, vector<pair<int, int>>,
                greater<pair<int, int>>>
    pq;

distances[source] = 0;
pq.push({0, source});
while (!pq.empty()) {
  auto [curr_distance, curr] = pq.top();
  pq.pop();
  if (distances[curr] < curr_distance) {
    continue;
  }
  for (const auto [weight, adj] : graph[curr]) {
    if (distances[adj] > distances[curr] + weight) {
      distances[adj] = distances[curr] + weight;
      parent[adj] = curr;
      pq.push({distances[adj], adj});
    }
  }
}
```

### All-pairs shortest paths

#### Floyd-Warshall

```cpp
// Floyd-Warshall
// Time complexity: O(V^3)
// Negative-weight cycles are detected.
// Negative-weight edges are ok.
vector<vector<int>> distances(N,
                              vector<int>(N, numeric_limits<int>::max() / 2));
for (int i = 0; i < N; ++i) {
  distances[i][i] = 0;
}
for (int i = 0; i < M; ++i) {
  int from, to, weight;
  cin >> from >> to >> weight;
  distances[from][to] = weight;
}

// parent[i][j]: the predecessor of vertex j on a shortest path from vertex i.
vector<vector<int>> parent(N, vector<int>(N, -1));
for (int i = 0; i < N; ++i) {
  for (int j = 0; j < N; ++j) {
    if (i != j && distances[i][j] != numeric_limits<int>::max() / 2) {
      parent[i][j] = i;
    }
  }
}

for (int k = 0; k < N; ++k) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (distances[i][j] > distances[i][k] + distances[k][j]) {
        // Updating in place is okay because it won't change if i or j is k.
        distances[i][j] = distances[i][k] + distances[k][j];
        parent[i][j] = parent[k][j];
      }
    }
  }
}

for (int i = 0; i < N; ++i) {
  if (distances[i][i] < 0) {
    // There is a negative-weight cycle in the graph.
  }
}
```

#### Johnson

### Maximum flow

### Minimum cost maximum flow

## Modulo operation

```cpp
-10 / 21 == 0
-10 % 21 == -10
```

- (Korean) <https://velog.io/@sw801733/%EB%82%98%EB%A8%B8%EC%A7%80-%EC%97%B0%EC%82%B0-%EB%B6%84%EB%B0%B0%EB%B2%95%EC%B9%99-%EB%AA%A8%EB%93%88%EB%9F%AC-%EC%97%B0%EC%82%B0>

```cpp
(A + B) % N == ((A % N) + (B % N)) % N
(A - B) % N == ((A % N) - (B % N)) % N
(A * B) % N == ((A % N) * (B % N)) % N
```

- How to check if A is a multiple of B without big integers.

```cpp
string A;
cin >> A;
int sum = 0;
for (auto it = A.cbegin(); it != c.end(); ++it) {
  sum = (sum * 10 + *it - '0') % B;
}
if (sum == 0) {
  cout << "A is a multiple of B.";
}
```

## Sweeping

## Greedy algorithm

## Dynamic programming

- 0-1 knapsack problem can be solved with an 1D array by iterating in the reverse order.

```cpp
vector<int> w(N + 1);
vector<int> v(N + 1);
vector<int> dp(K + 1);
for (int i = 1; i <= N; ++i) {
  for (int j = K; j - w[i] >= 0; --j) {
    dp[j] = max(dp[j], dp[j - w[i]] + v[i]);
  }
}
```

## Bit manipulation

```cpp
// Caution: be careful for the operator precedence.
if ((i & j) == 0 && (i ^ j) != 1) {
  // ...
}

// Caution: be careful for the data type.
1 << i; // It fails if i >= 32. Use (long long)1 instead.

a |= (1 << i);      // Set ith bit.
a &= ~(1 << i);     // Clear ith bit.
(a & (1 << i)) != 0 // Check ith bit.
(a & (1 << i)) == 1 // Bad: it may not be 1.
a = a & (a - 1);    // Clear the least significant bit.

// Iterate over all submask of a.
for (int i = a; i > 0; i = (i - 1) & a) {
  // ...
}

// Integer literal
int d = 42;
int o = 052;
int x = 0x2a;
int X = 0X2A;
int b = 0b101010; // Since C++14.
```

```cpp
#include <bitset>
bitset<n> b;                        // Initialize to 0.
bitset<n> b(u);                     // Initialize to `unsigned long long u`.
bitset<n> b(s, pos, m, zero, one);  // Initialize to string `s` from the index `pos` with `m` characters.
bitset<n> b(cp, pos, m, zero, one); // Initialize from the character array which is pointed by `cp`.

b.test(n);  // b[n];
b.set(n);   // b[n] = true;
b.reset(n); // b[n] = false;
b.flip(n);  // b[n] = !b[n];
```

- Caution: `bitset` has reversed indexes compared to `string`.
- If you need a variable length, not a fixed one, use `vector<bool>`.

## Binary search

```cpp
// Time complexity: O(logn)
int binary_search(const vector<int> &vec, int value) {
  int low = 0, high = vec.size() - 1;
  while (low <= high) {
    int mid = low + (high - low) / 2; // To avoid an overflow.
    if (value == vec[mid]) {
      return mid;
    } else if (value < vec[mid]) {
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  }
  return -1;
}

// Be careful when the value does not exist.
int lower_bound(const vector<int> &vec, int value) {
  int low = 0, high = vec.size() - 1;
  while (low <= high) {
    int mid = low + (high - low) / 2;
    if (value <= vec[mid]) {
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  }
  return low;
}

// Be careful when the value does not exist.
int upper_bound(const vector<int> &vec, int value) {
  int low = 0, high = vec.size() - 1;
  while (low <= high) {
    int mid = low + (high - low) / 2;
    if (value < vec[mid]) {
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  }
  return low;
}
```

```cpp
// binary_search: O(logn)
// A fully-sorted range can be used safely.
#include <algorithm>
template <class ForwardIt, class T>
constexpr bool binary_search(ForwardIt first, ForwardIt last, const T& value);
template <class ForwardIt, class T, class Compare>
constexpr bool binary_search(ForwardIt first, ForwardIt last, const T& value, Compare comp);

// lower_bound: O(logn)
// Find the first element such that value <= element.
// Find the first element such that element < value or comp(element, value) is false.
#include <algorithm>
template <class ForwardIt, class T>
constexpr ForwardIt lower_bound(ForwardIt first, ForwardIt last, const T& value);
template <class ForwardIt, class T, class Compare>
constexpr ForwardIt lower_bound(ForwardIt first, ForwardIt last, const T& value, Compare comp);

// upper_bound: O(logn)
// Find the first element such that value < element or comp(value, element) is true.
#include <algorithm>
template <class ForwardIt, class T>
constexpr ForwardIt upper_bound(ForwardIt first, ForwardIt last, const T& value);
template <class ForwardIt, class T, class Compare>
constexpr ForwardIt upper_bound(ForwardIt first, ForwardIt last, const T& value, Compare comp);

// `set` and `map` have their own member functions.
#include <set>
set<int> s;
auto it = s.lower_bound(key); // value_type: `Key`
auto it2 = s.upper_bound(key);

#include <map>
map<int> m;
auto it = m.lower_bound(key); // value_type: `std::pair<const Key, T>`
auto it2 = m.upper_bound(key);
```

### Longest increasing subsequence

- <https://leetcode.com/problems/longest-increasing-subsequence/>

```cpp
#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

vector<int> A(N);
for (int i = 0; i < N; ++i) {
  cin >> A[i];
}
vector<int> memo;
vector<int> length(N);
for (int i = 0; i < N; ++i) {
  auto it = lower_bound(memo.begin(), memo.end(), A[i]);
  if (it == memo.end()) {
    memo.push_back(A[i]);
    length[i] = memo.size();
  } else {
    *it = A[i];
    length[i] = it - memo.begin() + 1;
  }
}
int curr_length = memo.size();
vector<int> lis;
for (int i = N - 1; i >= 0 && curr_length > 0; --i) {
  if (length[i] == curr_length) {
    lis.push_back(A[i]);
    --curr_length;
  }
}
for (auto it = lis.crbegin(); it != lis.crend(); ++it) {
  cout << *it << ' ';
}
```

## Sliding window

## Two pointers

## Cycle detection

- <https://en.wikipedia.org/wiki/Cycle_detection>
- <https://leetcode.com/problems/find-the-duplicate-number/description/>

```cpp
// Find a repetition x_i = x_2i.
Node *slow = head;
Node *fast = head;
bool has_cycle = false;
while (fast != nullptr && fast->next != nullptr) {
  slow = slow->next;
  fast = fast->next->next;
  if (slow == fast) {
    // At this point, the distance between them will be
    // divisible by the period λ.
    has_cycle = true;
    break;
  }
}

// Check if there is a cycle.
if (!has_cycle) {
  // ...
}

// Find the position μ of first repetition.
// Two pointers will agree as soon as the slow one reaches index μ.
int mu = 0;
slow = head;
while (slow != fast) {
  slow = slow->next;
  fast = fast->next;
  ++mu;
}

// Find the length of the shortest cycle starting from x_μ.
int length = 1;
fast = slow->next;
while (slow != fast) {
  fast = fast->next;
  ++length;
}
```

## Backtracking

## Prefix sum

## Offline queries

## Coordinate compression

- (Korean) <https://stonejjun.tistory.com/136>

```cpp
vector<ll> v;
ll arr[1010101];

int main() {
  for (i = 1; i <= n; i++) {
    scanf("%lld", &arr[i]);
    v.push_back(arr[i]);
  }
  v.push_back(-inf);
  sort(v.begin(), v.end());
  v.erase(unique(v.begin(), v.end()), v.end());
  for (i = 1; i <= n; i++) {
    arr[i] = lower_bound(v.begin(), v.end(), arr[i]) - v.begin();
  }
}
```

## Line intersection

```cpp
int ccw(pair<int, int> p1, pair<int, int> p2, pair<int, int> p3) {
  int s = p1.first * p2.second + p2.first * p3.second + p3.first * p1.second;
  s -= p2.first * p1.second + p3.first * p2.second + p1.first * p3.second;
  if (s > 0) {
    return 1;
  }
  if (s < 0) {
    return -1;
  }
  return 0;
}
```

- CCW
  - $p_1(x_1,y_1), p_2(x_2,y_2), p_3(x_3,y_3)$
  - $S = det\begin{pmatrix} x_2-x_1 & y_2-y_1 \\ x_3-x_2 & y_3-y_2 \end{pmatrix} \\ = (x_1y_2 + x_2y_3 + x_3y_1) - (x_2y_1 + x_3y_2 + x_1y_3)$
  - S > 0: Counterclockwise
  - S < 0: Clockwise
  - S = 0: Linear
- GitHub cannot render the matrix properly, so I have attached an alternative version of the equation above.

```math
S = det\begin{pmatrix} x_2-x_1 & y_2-y_1 \\ x_3-x_2 & y_3-y_2 \end{pmatrix} \\ = (x_1y_2 + x_2y_3 + x_3y_1) - (x_2y_1 + x_3y_2 + x_1y_3)
```

```cpp
bool lines_intersect(pair<int, int> p1, pair<int, int> p2, pair<int, int> p3,
                     pair<int, int> p4) {
  int p1p2 = ccw(p1, p2, p3) * ccw(p1, p2, p4);
  int p3p4 = ccw(p3, p4, p1) * ccw(p3, p4, p2);
  if (p1p2 == 0 && p3p4 == 0) {
    if (p1 > p2) {
      swap(p1, p2);
    }
    if (p3 > p4) {
      swap(p3, p4);
    }
    return p3 <= p2 && p1 <= p4;
  }
  return p1p2 <= 0 && p3p4 <= 0;
}
```

- (Korean) <https://killerwhale0917.tistory.com/6>
- (Korean) <https://jordano-jackson.tistory.com/27>
- (Korean) <https://nahwasa.com/entry/CCW%EB%A5%BC-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EC%84%A0%EB%B6%84-%EA%B5%90%EC%B0%A8-%ED%8C%90%EC%A0%95>

## Primality test

```cpp
// Time complexity: O(N^(1/2))
bool is_prime(int N) {
  if (N < 2) {
    return false;
  }
  for (int i = 2; i <= sqrt(N); ++i) {
    if (N % i == 0) {
      return false;
    }
  }
  return true;
}
```

```cpp
// Sieve of Eratosthenes
// Time complexity: O(NloglogN)
vector<bool> sieve(N + 1, true);
sieve[0] = false;
sieve[1] = false;
for (int i = 2; i <= sqrt(N); ++i) {
  if (!sieve[i]) {
    continue;
  }
  for (int j = i * i; j <= N; j += i) {
    sieve[j] = false;
  }
}
```

## Brute force

## Overflow checking

- <https://stackoverflow.com/questions/199333/how-do-i-detect-unsigned-integer-overflow>

```cpp
#include <limits>
if (x > 0 && a > numeric_limits<int>::max() - x)  // a + x would overflow.
if (x < 0 && a < numeric_limits<int>::min() - x)  // a + x would underflow.
if (x < 0 && a > numeric_limits<int>::max() + x)  // a - x would overflow.
if (x > 0 && a < numeric_limits<int>::min() + x)  // a - x would underflow.
if (a == -1 && x == numeric_limits<int>::min())   // a * x can overflow.
if (x == -1 && a == numeric_limits<int>::min())   // a * x can overflow.
if (x != 0 && a > numeric_limits<int>::max() / x) // a * x would overflow.
if (x != 0 && a < numeric_limits<int>::min() / x) // a * x would underflow.
```

## C++ standard library & syntax

### `iostream`

```cpp
#include <iostream>
cin.tie();             // Return the pointer to the connected stream.
cin.tie(&cout);        // Connect `cin` to `cout` (only one connection can be made at a time).

// Fast I/O
cin.tie(nullptr);      // Disconnect.
ios_base::sync_with_stdio(false); // Separate buffers of C and C++. The simultaneous use of C and C++ IO functions is prohibited.
// Use `'\n'` instead of `endl` for the additional performance improvement.
```

```cpp
// Formatting functions
#include <iostream>
cout.width(n);     // Set the field width (applied only to the next output).
cout.fill('*');    // Fill the remaining field width with the specified character (kept until changed).
cout.precision(n); // Round and output with a total of n significant digits (default) or n decimal places (for fixed, scientific, and hexfloat formats).

#include <iomanip> // Same as above, but requires the <iomanip> header.
cout << setw(n);
cout << setfill('*');
cout << setprecision(n);
cout << setbase(b); // Output an integer in base b.

// Save and restore cout formatting.
ios_base::fmtflags f(cout.flags());
// Your code here...
cout.flags(f);

// Input/output manipulators
// cout.setf(ios_base::showpoint); == cout << showpoint;
// cout.unsetf(ios_base::showpoint);
boolalpha, noboolalpha // Input and output bool values as true and false / 1 and 0 (must be separated by spaces).
showbase, noshowbase   // Show / hide the base prefix notation.
showpoint, noshowpoint // Always display the trailing decimal point / Display only when there is a fractional part.
showpos, noshowpos     // Show / hide the + sign in front of non-negative numbers.
uppercase, nouppercase // Display specific characters in uppercase / lowercase (e.g. hexadecimal, scientific E notation, etc.).
dec
oct
hex                    // If dec, oct, and hex are all passed to unsetf, recognize the format based on input.
left
right
internal               // Insert a space between the sign and the value.
fixed                  // Use fixed-point notation.
scientific             // Use scientific E notation.
hexfloat
defaultfloat
unitbuf, nounitbuf     // Flush output without buffering.
skipws, noskipws       // Ignore whitespace (blank, tab, newline, formfeed, carriage return) during input.
flush
ends                   // Append null and flush.
endl                   // Append newline and flush.
basefield
adjustfield
floatfield
```

### `algorithm`

```cpp
#include <algorithm>
max(a, b);
max({1, 2, 3});
min(a, b);
min({1, 2, 3});
auto it = max_element(s.cbegin(), s.cend());
auto it2 = min_element(s.cbegin(), s.cend());

void sort(RandomIt first, RandomIt last); // O(nlogn)
void sort(RandomIt first, RandomIt last, Compare comp);
sort(s.begin(), s.end());
sort(s.begin(), s.end(), greater<int>());
sort(arr, arr + 3);

void partial_sort(RandomIt first, RandomIt middle, RandomIt last); // O(nlogm)
void partial_sort(RandomIt first, RandomIt middle, RandomIt last, Compare comp);

void nth_element(RandomIt first, RandomIt nth, RandomIt last); // Average: O(n)
void nth_element(RandomIt first, RandomIt nth, RandomIt last, Compare comp);

void reverse(BidirIt first, BidirIt last); // O(n)

bool next_permutation(BidirIt first, BidirIt last);
bool next_permutation(BidirIt first, BidirIt last, Compare comp);
bool prev_permutation(BidirIt first, BidirIt last);
bool prev_permutation(BidirIt first, BidirIt last, Compare comp);
bool is_permutation(ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2);
bool is_permutation(ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2, ForwardIt2 last2);
```

### `functional`

```cpp
#include <functional>
template <class T = void>
struct greater;           // x > y
template <class T = void>
struct less;              // x < y
```

### `pair`

```cpp
#include <utility>
template <class T1, class T2>
struct pair;
template <class T1, class T2>
constexpr std::pair<T1, T2> make_pair(T1&& t, T2&& u);

pair<T1, T2> p(v1, v2); // Value initialization if the constructor (explicit) is omitted.
make_pair(v1, v2);      // The type is inferred by v1 and v2.
p.first, p.second;      // Member access.
{v1, v2};               // Brace initialization is possible for the argument.
tie(x,y) = p;           // Destructure after converted to tuple automatically.
auto [x, y] = t;        // Structured binding. Since C++17.
// Relational, ==, != operations are possible.
```

### `tuple`

- (Korean) <https://jjeongil.tistory.com/148>

```cpp
#include <tuple>
tuple<T1, T2, ..., Tn> t(v1, v2, ..., vn); // Value initialization if the constructor (explicit) is omitted.
make_tuple(v1, v2, ..., vn);
get<i>(t);                         // ith member of t (lvalue reference if t is lvalue, rvalue reference otherwise).
{v1, v2, ..., vn};                 // Brace initialization is possible for the argument.
tie(x,y,z) = t;                    // Destructure t as x, y, z.
auto [x, y, z] = t;                // Structured binding. Since C++17.
tuple_size<tupleType>::value;      // The number of members of tupleType.
tuple_element<i, tupleType>::type; // The type of ith member of tupleType.
```

### `iterator`

```cpp
#include <iterator>
auto it2 = next(it, n = 1); // O(n)
auto it3 = prev(it, n = 1); // O(n)

vector<int> v{3, 1, 4};
distance(v.begin(), v.end()) == 3 // O(1) for `LegacyRandomAccessIterator`, O(n) otherwise.
```

### `numeric`

```cpp
#include <numeric>
accumulate(vec.begin(), vec.end(), 0); // Calculates the sum of the elements in `vec`.
accumulate(first, last, init, binary_op);
reduce(first, last, init, binary_op); // Similar to `accumulate`, except that `reduce` may be applied out of order.
gcd(M m, N n); // return abs(m == 0 ? n : gcd(n % m, m));
lcm(M m, N n);
```

### `cmath`

```cpp
#include <cmath>
abs(num);
sqrt(num);
ceil(num);
floor(num);
round(num); // It rounds halfway cases away from zero (e.g. -0.5 -> -1).
log(num); // log(e)(num)

// Round from the nth digit below the decimal point.
num = round(num * pow(10, n-1)) / pow(10, n-1);
```

### `swap`

```cpp
#include <utility>
template <class T>
constexpr void swap(T& a, T& b) noexcept; // O(1)
```

### `random`

```cpp
#include <random>
random_device rd; // It can be expensive based on the implementation. Use to generate seed only.
default_random_engine e(rd()); // It is a pseudo random and relatively cheap.
e(); // It returns a random integer.
e.seed(rd()); // It can be seeded after initialized.
```

### `struct`

```cpp
struct Point {
  int x;
  int y;

  Point() : x(0), y(0) {}
  ~Point() {}

  bool operator<(const Point& other) const {
    if (x != other.x) {
      return x < other.x;
    }
    return y < other.y;
  }
}; // There should be a semicolon.

inline bool operator==(const Point &lhs, const Point &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y;
}

inline bool operator!=(const Point &lhs, const Point &rhs) {
  return !(lhs == rhs);
}

Point point{10, 2}; // Uniform initialization (x == 10, y == 2).
Point *point = new Point(); // Dynamic allocation.
delete point;
```

### `enum`

```cpp
// Unscoped enumeration.
enum Color {red, green, blue};
Color eyes = green; // Implicitly use `Color::green`.
Color hair = Color::red; // Explicitly use `Color::red`.

// Scoped enumeration (you can use `struct` instead of `class`).
enum class Color2 {red, green, blue};
Color2 p = Color2::red; // Should use `Color2::red` explicitly.

// Unnamed, unscoped enumeration.
enum {red, green, blue};
```

## Sorting algorithms

### Insertion sort

### Merge sort

```cpp
#include <vector>
using namespace std;

vector<int> arr(N);
vector<int> temp(N);
merge_sort(arr, 0, N - 1, temp);

void merge_sort(vector<int> &arr, int left, int right, vector<int> &temp) {
  if (left >= right) {
    return;
  }
  int mid = left + (right - left) / 2;
  merge_sort(arr, left, mid, temp);
  merge_sort(arr, mid + 1, right, temp);
  merge(arr, left, mid, right, temp);
}

void merge(vector<int> &arr, int left, int mid, int right,
            vector<int> &temp) {
  for (int i = left; i <= right; ++i) {
    temp[i] = arr[i];
  }

  int i = left, j = mid + 1, k = left;
  while (i <= mid && j <= right) {
    if (temp[i] <= temp[j]) {
      arr[k] = temp[i];
      ++i;
    } else {
      arr[k] = temp[j];
      ++j;
    }
    ++k;
  }

  while (i <= mid) {
    arr[k] = temp[i];
    ++i;
    ++k;
  }
  while (j <= right) {
    arr[k] = temp[j];
    ++j;
    ++k;
  }
}
```

### Heapsort

```cpp
#include <algorithm>

// make_heap
// Time complexity: O(n)
template <class RandomIt>
constexpr void make_heap(RandomIt first, RandomIt last);
template <class RandomIt, class Compare>
constexpr void make_heap(RandomIt first, RandomIt last, Compare comp);

// sort_heap
// Time complexity: O(nlogn)
// [first, last) should be a heap.
template <class RandomIt>
constexpr void sort_heap(RandomIt first, RandomIt last);
template <class RandomIt, class Compare>
constexpr void sort_heap(RandomIt first, RandomIt last, Compare comp);

// pop_heap
// Time complexity: O(logn)
// Swaps the value at first and the value at last-1.
// Makes [first, last-1) into a heap.
template <class RandomIt>
constexpr void pop_heap(RandomIt first, RandomIt last);
template <class RandomIt, class Compare>
constexpr void pop_heap(RandomIt first, RandomIt last, Compare comp);

// push_heap
// Time complexity: O(logn)
// inserts the element at last-1 into [first, last-1)
template <class RandomIt>
constexpr void push_heap(RandomIt first, RandomIt last);
template <class RandomIt, class Compare>
constexpr void push_heap(RandomIt first, RandomIt last, Compare comp);

// is_heap
// Time complexity: O(n)
template <class RandomIt>
constexpr bool is_heap(RandomIt first, RandomIt last);
template <class RandomIt, class Compare>
constexpr bool is_heap(RandomIt first, RandomIt last, Compare comp);

// is_heap_until
// Time complexity: O(n)
template <class RandomIt>
constexpr RandomIt is_heap_until(RandomIt first, RandomIt last);
template <class RandomIt, class Compare>
constexpr RandomIt is_heap_until(RandomIt first, RandomIt last, Compare comp);
```

### Quicksort

### Counting sort

### Radix sort

### Bucket sort
