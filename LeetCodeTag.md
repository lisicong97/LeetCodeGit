看清楚题目，看看test case

解决子问题-递归，这一步决策和之前的决策有关系-递归 + cache = dp

连续的list考虑slide_window 

不连续考虑dp

dfs/bfs BFS可以考虑PriorityQueue

一个一个凑可以用binary search

inorder postorder preorder

发现没什么可以套，找找规律，看看有什么数学方法

preSum用hashMap存时，key为preSum，value为出现次数，注意map初始化后，map[0] = 1

最小修改使得s1 == s2的dp题目记得初始化dp[ 0 ] [ i ] 和 dp[ i ] [ 0 ]

时间复杂度
T(n) = a*T(n/b)+ n^k
    if (a > b^k)   T(n) = O(n^(logb(a)));
    if (a = b^k)   T(n) = O(n^k * logn);
    if (a < b^k)   T(n) = O(n^k); 

Dfs bfs time complexity: map: O(N+E) matrix:O(N^2)

# sort

```Java
插入排序
public void insertSort(int[] a){
    int len=a.length;
    for(int i=1;i<len;i++){
        int insertVal=a[i];    //待插入元素
        int index=i-1;         //插入的位置（准备和前一个作比较）
        while(index>=0 && insertVal<a[index]){
            a[index+1]=a[index];
            index--;
        }
        a[index+1]=insertVal;      //把插入的数放到合适的位置
    }
}
希尔排序
public void shellSort(int[] a){
    int dk=a.length/2;
    while(dk>=1){
        ShellInsertSort(a,dk)
        dk=dk/2;
    }
}
public void ShellInsertSort(int[] a,int dk){
    //类似插入排序，只是插入排序增量是1，这里增量是dk
    for(int i=dk;i<a.length;i++){
        if(a[i]<a[i-dk]){
            int j;
            int x=a[i];   //x为待插入元素   
            a[i]=a[i-dk];
            for(j=i-dk;j>=0&&x<a[j];j=j-dk){  
            //通过循环，逐个后移一位找到要插入的位置。
                a[j+dk]=a[j];
            }
            a[j+dk]=x;   //插入
        }
    }
}
冒泡排序
public void bubbleSort(int[] a, int n ){
    int i,j;
    for(i=0; i<n; i++){
        for(j=1; j<n-i; j++){
            if(a[j-1]>a[j]){
                int temp;
                temp=a[j-1];
                a[j-1]=a[j];
                a[j]=temp;
            }   
        }
    }   
}
快速排序
public void QuickSort(int[] a, int low, int high){
    int temp;
    int i=low, j=high;
    if(low<high){
        temp=a[low];
        while(i!=j){
            while(i<j && a[j]>temp) j--;
            if(i<j){
                a[i]=a[j];
                i++;
            }
            while(i<j && a[i]<temp) i++;
            if(i<j){
                a[j]=a[i];
                j--;
            }           
        }
        a[i]=temp;
        QuickSort(a, low, i-1);
        QuickSort(a, i+1, high);
    }   
}
堆排序
//因R[]中是一颗完全二叉树，所以元素的存储必须从1开始，
    public static void Shift(int[] R,int low, int high){
        int i=low, j=2*i; //R[j]是R[i]的左孩子结点
        int temp=R[i];
        while(j<=high){
            if(j<high && R[j]<R[j+1])
                j++;
            if(temp<R[j]){   //如果当前值小于孩子结点，
                R[i]=R[j];   //则交换后继续往下走
                i=j;
                j=2*i;
            }
            else
                break;
        }
        R[i]=temp;
    }
    /*堆排序函数*/
    public static void heapSort(int[] R, int n){
        int i;
        int temp;
        for(i=n/2;i>0;--i)    //建立初始堆
            Shift(R,i,n);
        for(i=n;i>1;--i){     //进行n-1次循环完成堆排序
            temp=R[1];
            R[1]=R[i];
            R[i]=temp;
            Shift(R,1,i-1);
        }
    }
归并排序
public static void sort(int[] arr, int L, int R) {
    if(L == R) {
        return;
    }
    int mid = L + ((R - L) >> 1);
    sort(arr, L, mid);
    sort(arr, mid + 1, R);
    merge(arr, L, mid, R);
}

public static void merge(int[] arr, int L, int mid, int R) {
    int[] temp = new int[R - L + 1];
    int i = 0;
    int p1 = L;
    int p2 = mid + 1;
    // 比较左右两部分的元素，哪个小，把那个元素填入temp中
    while(p1 <= mid && p2 <= R) {
        temp[i++] = arr[p1] < arr[p2] ? arr[p1++] : arr[p2++];
    }
    // 上面的循环退出后，把剩余的元素依次填入到temp中
    // 以下两个while只有一个会执行
    while(p1 <= mid) {
        temp[i++] = arr[p1++];
    }
    while(p2 <= R) {
        temp[i++] = arr[p2++];
    }
    // 把最终的排序的结果复制给原数组
    for(i = 0; i < temp.length; i++) {
        arr[L + i] = temp[i];
    }
}

基数排序
    int exp = 1; // 1, 10, 100, 1000 ...
    int R = 10; // 10 digits

    int[] aux = new int[nums.length];
    
    while (m / exp > 0) { // Go through all digits from LSB to MSB
        int[] count = new int[R];
        
        for (int i = 0; i < nums.length; i++) {
            count[(nums[i] / exp) % 10]++;
        }
        
        for (int i = 1; i < count.length; i++) {
            count[i] += count[i - 1];
        }
        
        for (int i = nums.length - 1; i >= 0; i--) {
            aux[--count[(nums[i] / exp) % 10]] = nums[i];
        }
        
        for (int i = 0; i < nums.length; i++) {
            nums[i] = aux[i];
        }
        exp *= 10;
    }

```

# Postorder inorder preorder

```
iterative 中序遍历
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> list = new ArrayList<Integer>();
        Stack<TreeNode> stack = new Stack<TreeNode>();
        TreeNode cur = root;

        while(cur!=null || !stack.empty()){
            while(cur!=null){
                stack.add(cur);
                cur = cur.left;
            }
            cur = stack.pop();
            list.add(cur.val);
            cur = cur.right;
        }
        
        return list;
    }
iterative 前序遍历
	public List<Integer> preorderTraversal(TreeNode node) {
		List<Integer> list = new LinkedList<Integer>();
		Stack<TreeNode> rights = new Stack<TreeNode>();
		while(node != null) {
			list.add(node.val);
			if (node.right != null) {
				rights.push(node.right);
			}
			node = node.left;
			if (node == null && !rights.isEmpty()) {
				node = rights.pop();
			}
		}
	    return list;
	}
def preorderTraversal(self, root):
    ret = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node:
            ret.append(node.val)
            stack.append(node.right)
            stack.append(node.left)
    return ret
    
iterative 后序遍历
	public List<Integer> postorderTraversal(TreeNode root) {
		LinkedList<Integer> ans = new LinkedList<>();
		Stack<TreeNode> stack = new Stack<>();
		if (root == null) return ans;
		
		stack.push(root);
		while (!stack.isEmpty()) {
			TreeNode cur = stack.pop();
			ans.addFirst(cur.val);
			if (cur.left != null) {
				stack.push(cur.left);
			}
			if (cur.right != null) {
				stack.push(cur.right);
			} 
		}
		return ans;
	}
```



# median

**295 一个数一个数得加进list，随时需要能拿到它的median**

一个最小堆 11、31、44……一个最大堆 5、4、1…… 每来一个数，一个堆offer、poll，一个堆offer，保持a .size == b.size or a.size + 1 == b.size

**480 滑动窗口求median**

同上，每次移动，去掉左端数字，加入右端数字（一个堆offer、poll，一个堆offer）

**4 Median of Two Sorted Arrays**

用二分法找最长那个list的点。其实思想也是分成有序的两堆，找各自的顶点。

# 二分查找

```java
while (start <= end) {
	  mid = start + (end - start) / 2;
		if (arr[mid] < key)
			  start = mid + 1;
		else if (arr[mid] > key)
			  end = mid - 1;
		else {
			  ret = mid;  
        break;
    }
}
```

**69 Sqrt(x)**

```Java
public int sqrt(int x) {
    if (x == 0)
        return 0;
    int left = 1, right = Integer.MAX_VALUE;
    while (true) {
        int mid = left + (right - left)/2;
        if (mid > x/mid) {
            right = mid - 1;
        } else {
            if (mid + 1 > x/(mid + 1))
                return mid;
            left = mid + 1;
        }
    }
}
```

**33 Search in Rotated Sorted Array**

list被左移了，先找pivot

```Java
while (start < end) { // start == end 就没有找的必要了
    int mid = start + (end -  start) / 2;
    // mid应该与end比，因为我们要找最小的，参考[1，2，3]、[3，1]
    if (nums[mid] > nums[end]) start = mid + 1; 
    else end = mid; // 不是mid - 1，mid有可能是pivot
}
return start;
```

**81 Search in Rotated Sorted Array II**

https://leetcode.com/submissions/detail/292570744/ 开始有duplicate了。

也是while(start<=end). start = mid + 1 or end = mid - 1。

不过这里得先判断哪边是sorted的（nums[start] < nums[mid]），然后判断要去rotated那边还是sorted那边。

不巧相等的话start++

**34 Find First and Last Position of Element in Sorted Array**

如果找最左边的

```java
while(start<=end) {
    int mid = (start+end)/2;
    if(nums[mid]>=target) end=mid-1; // 就算等于target，end也左移
    else start=mid+1;
    if(nums[mid] == target) 
        res = mid; // 暂时存一下，有可能是正确答案
}
```

**378 Kth Smallest Element in a Sorted Matrix**

一个倾斜矩阵，找到最小的第K个。

这种矩阵都有二叉搜索树的性质。binary search是可以把答案从matrix[0] [0] 到 matrix[size-1] [size-1]二分找到那个值。

```java
public class Solution {
    public int kthSmallest(int[][] matrix, int k) {
        int lo = matrix[0][0], hi = matrix[matrix.length - 1][matrix[0].length - 1] + 1;//[lo, hi)
        while(lo < hi) {
            int mid = lo + (hi - lo) / 2;
            int count = 0,  j = matrix[0].length - 1;
            for(int i = 0; i < matrix.length; i++) {
                while(j >= 0 && matrix[i][j] > mid) j--;
                count += (j + 1);
            }
            if(count < k) lo = mid + 1;
            else hi = mid; // 如果count == k, 有可能是正确答案，hi不能 = mid - 1
        }
        return lo;
    }
}
```

300 Longest Increasing Subsequence https://leetcode.com/problems/longest-increasing-subsequence/discuss/74824/JavaPython-Binary-search-O(nlogn)-time-with-explanation

```Java
public int lengthOfLIS(int[] nums) {
    int[] tails = new int[nums.length];
    int size = 0;
    // (1) if x is larger than all tails, append it, increase the size by 1
    // (2) if tails[i-1] < x <= tails[i], update tails[i]
    for (int x : nums) {
        int i = 0, j = size;
        while (i != j) {
            int m = (i + j) / 2;
            if (tails[m] < x)
                i = m + 1;
            else
                j = m;
        }
        tails[i] = x;
        if (i == size) ++size;
    }
    return size;
}
```



# Binary Search Tree

**98 Validate Binary Search Tree**

判断一棵树是不是平衡搜索树，可以用inorder看是不是sorted的，也可以用递归，判断点的范围是否在参数规定内。

**449 Serialize and Deserialize BST**

preorder存下来，deserialize的时候，如果点的范围不在参数规定内，return null。因为preoder的话，队列里的下一个node必然是左子树or右子树，

https://leetcode.com/problems/serialize-and-deserialize-bst/discuss/177617/the-General-Solution-for-Serialize-and-Deserialize-BST-and-Serialize-and-Deserialize-BT

**297 Serialize and Deserialize Binary Tree**

preorder存的时候，null存为#，deserialize的时候poll出#直接return。

**99 Recover Binary Search Tree**

找俩错位的点，inorder保存前一个，如果prev大于current，这俩点都有问题，prev只需要找一次，current一直找。

**235, 236找树上俩节点的最低公祖先**

最低公共祖先，包括binary tree和binary search tree

Binary tree

```Java
public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    if (root == null || root == p || root == q) return root;
    TreeNode left = lowestCommonAncestor(root.left, p, q);
    TreeNode right = lowestCommonAncestor(root.right, p, q);
    if(left == null && right == null) return null; 
    if(left != null && right != null) return root;  
    return left == null ? right : left;
}
```

BST

dfs，找到满足root的value大于small小于big，root比small还小了找root右边，不然找左边。

# Tree Recursion

**652 Find Duplicate Subtrees**

有没有子树是完全一样的结构，考虑序列化之后比较。用postorder，这样子树的key都已经知道，再与本node比较。

# meeting room interval

56, https://www.cnblogs.com/grandyang/p/5244720.html

interval maximium
每个range按结束时间sort，遍历，看能不能加进res

# 图形求面积

**211 Maximal Square**

用dynamical programming，看看这个点的[i] [j-1]和[i-1] [j]和[i-1] [j-1]有无正方形，选最小的 + 1

**84 Largest Rectangle in Histogram**

一行扫描过去，每个index都作为height，看左边和右边能延伸到哪里。

```java
for (int i = 1; i < height.length; i++) {
    int p = i - 1;
    while (p >= 0 && height[p] >= height[i]) {
        p = lessFromLeft[p];
    }
    lessFromLeft[i] = p;
}
...
maxArea = Math.max(maxArea, height[i] * (lessFromRight[i] - lessFromLeft[i] - 1));
```

**85 Maximal Rectangle**

每一行都转换成包括前几行记录的柱状图，然后使用柱状图求面积的方法

**11 Container With Most Water**

左右两端同时扫描，与最大的比较，if (height[leftIndex] < height[rightIndex]) leftIndex++ else rightIndex--

# 滑动窗口

找序列中符合条件的子序列用滑动窗口

**76 Given a string S and a string T, find the minimum window in S which will contain all the characters in T**

**992 Subarrays with K Different Integers**

变成atMost(K) - atMost(K-1)

# DP

**72 最短编辑距离**

```java
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            if(word1.charAt(i) == word2.charAt(j))
                cost[i + 1][j + 1] = cost[i][j];
            else {
                int a = cost[i][j];
                int b = cost[i][j + 1];
                int c = cost[i + 1][j];
                cost[i + 1][j + 1] = min(a,b,c)+1;
            }
        }
    }
```

**516 Longest Palindromic Subsequence**

一个string删去几个char得到的最长对称sequence，dp [i] [j] = if (s[i] == s[j]) dp [i+1] [j-1] + 2 else Math.max(dp [i+1] [j], dp [i] [j-1])



**188 买股票**

```
    public int maxProfit(int k, int[] prices) {
        int len = prices.length;
        
        int[][] t = new int[k + 1][len];
        for (int i = 1; i <= k; i++) {
            int tmpMax =  -prices[0];
            for (int j = 1; j < len; j++) {
                t[i][j] = Math.max(t[i][j - 1], prices[j] + tmpMax);
                tmpMax =  Math.max(tmpMax, t[i - 1][j - 1] - prices[j]);
            }
        }
        return t[k][len - 1];
    }
```

**2008 Maximum Earnings From Taxi**

dp[i] = dp[i-1] temporary

然后根据已有条件反推 dp[i] = dp[s] + func(s~i)

# Stack

**155 Min stack**

除了stack的基本功能，需要有一个能直接返回stack中最小的元素的method。

stack中存个class Node(val value: Int, var next: Node?, val min: Int) 每push一次，在Node里存此时最小值。

**716 Max stack**

除了在stack能找到max，还要能把它pop出来。

建另一个stack2用于存储max，push的时候看看要不要进入stack2，pop的时候看看stack2要不要pop

popMax的时候，建一个temStack，stack1一个一个找，找到了max给他移出去，再把临时存在temStack的塞回去。

**739 Daily Temperatures**

一个array，找下一个比他大的，没有的话为0

```kotlin
        val stack = Stack<Int>()
        val res = IntArray(temperatures.size) { 0 }
        for (i in temperatures.indices) {
            while (stack.isNotEmpty() && temperatures[stack.peek()] < temperatures[i]) {
                val index = stack.pop()
                res[index] = i - index
            } 
            stack.push(i)
        }
        return res
```



# Parentheses

()匹配数量的计算方法是C(n,2n) - C(n-1,2n)，卡特兰数

**20 valid Parentheses**

检查是否有效，用个stack

**22 Generate Parentheses**

递归，记录左括号和右括号个数

**32 Longest Valid Parentheses**

dp

```kotlin
        val V = Array<Int>(s.length){0}
        var open = 0
        var max = 0
        for (i in 0 until s.length) {
            if (s[i] == '(') {
                open++
            } else if (open > 0) {
                V[i] = V[i-1] + 2
                if (i - V[i] > 0) {
                    V[i] += V[i - V[i]]
                }
                open--
            }
            if (V[i] > max) {
                max = V[i]
            }
        }
        return max
```



**678 Valid Parenthesis String**

带有*，可以代替任何字符

```Java
    public boolean checkValidString(String s) {
        int low = 0; // 未配对左括号最小个数
        int high = 0; // 未配对左括号最大个数
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                low++;
                high++;
            } else if (s.charAt(i) == ')') {
                if (low > 0) {
                    low--;
                }
                high--;
            } else { // *
                if (low > 0) { // * == ）
                    low--;
                }
                high++;
            }
          // 这句话是必须的，扫描的中途可能出错
            if (high < 0) {
                return false;
            }
        }
        return low == 0;
    }
```

**301 Remove Invalid Parentheses**

删几个char，让paren合理。

```java
public void remove(String s, List<String> ans, int last_i, int last_j,  char[] par) {
    for (int stack = 0, i = last_i; i < s.length(); ++i) {
        if (s.charAt(i) == par[0]) stack++;
        if (s.charAt(i) == par[1]) stack--;
        if (stack >= 0) continue;
        for (int j = last_j; j <= i; ++j)
            if (s.charAt(j) == par[1] && (j == last_j || s.charAt(j - 1) != par[1]))
                remove(s.substring(0, j) + s.substring(j + 1, s.length()), ans, i, j, par);
        return;
    }
    String reversed = new StringBuilder(s).reverse().toString();
    if (par[0] == '(') // finished left to right
        remove(reversed, ans, 0, 0, new char[]{')', '('});
    else // finished right to left
        ans.add(reversed);
}
```

# 快慢指针

**287 Find the Duplicate Number**

一个二倍速指针和一个普通指针，相遇后有x（进入circle前的len） = z（相遇点前进到切点）的性质

https://leetcode.com/problems/find-the-duplicate-number/discuss/72846/My-easy-understood-solution-with-O(n)-time-and-O(1)-space-without-modifying-the-array.-With-clear-explanation.

# MASK

**2002 Maximum Product of the Length of Two Palindromic Subsequences**

用二进制mask获取subsequence mask = 1 << s.length - 1, all res in 1 .. mask

# 杂项

图的聚合

    vector<int> parent;     
    int findParent(int node) {
        return parent[node] == node ? node : findParent(parent[node]);
    }    
    void makeSameGroup(int u , int v) {
        int pu = findParent(u) ; 
        int pv = findParent(v);
        parent[pv] = pu; 
    }
    
    bool validPath(int n, vector<vector<int>>& edges, int start, int end) {
        parent.resize(n);
        for(int i=0;i<n;i++)
            parent[i] = i; 
        
        for(auto e : edges) {
            makeSameGroup(e[0] , e[1]); 
        }                         
        return findParent(start) == findParent(end); 
    }

return结果太大，要求取模拿特征时dp[i] = (... + **m**) % m 因为有可能...中dp[j] - dp[k] < 0

辗转相除法
public int GCD(int a, int b){
    while(b != 0 ){
        int temp = b;
        b = a%b;
        a = temp;
    }
    return a;
}

procedure three-way-partition: 荷兰旗三分法
    i ← 0 j ← 0 k ← size of A
    while j < k:
        if A[j] < mid:
            swap A[i++] and A[j++]
        else if A[j] > mid:
            k ← k - 1
            swap A[j] and A[k]
        else:
            j ← j + 1

224 加减法括号计算
public int calculate(String s) {
    Stack<Integer> stack = new Stack<Integer>();
    int result = 0;
    int number = 0;
    int sign = 1;
    for(int i = 0; i < s.length(); i++){
        char c = s.charAt(i);
        if(Character.isDigit(c)){
            number = 10 * number + (int)(c - '0');
        }else if(c == '+'){
            result += sign * number;
            number = 0;
            sign = 1;
        }else if(c == '-'){
            result += sign * number;
            number = 0;
            sign = -1;
        }else if(c == '('){
            //we push the result first, then sign;
            stack.push(result);
            stack.push(sign);
            //reset the sign and result for the value in the parenthesis
            sign = 1;   
            result = 0;
        }else if(c == ')'){
            result += sign * number;  
            number = 0;
            result *= stack.pop();    //stack.pop() is the sign before the parenthesis
            result += stack.pop();   //stack.pop() now is the result calculated before the parenthesis
            }
}
result += sign * number;
return result;



}
加减乘除无括号
    public int calculate(String s) {
        if (s == null || s.length() == 0) return 0;
        Stack<Integer> stack = new Stack<>();
        s += '+';
        char op = '+';
        for (int i = 0, n = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c >= '0' && c <= '9') { n = n * 10 + c - '0'; continue; }
            if (c == ' ') continue;
            if (op == '+') stack.push(n);
            else if (op == '-') stack.push(-n);
            else if (op == '\*') stack.push(stack.pop()*n);
            else if (op == '/') stack.push(stack.pop()/n);
            op = c;
            n = 0;
        }

​	int total = 0;
​	while (!stack.isEmpty()) total += stack.pop();
​	return total;

}

加减乘除有括号

```
    int calculate(string s) {
        int n = s.size(), num = 0, curRes = 0, res = 0;
        char op = '+';
        for (int i = 0; i < n; ++i) {
            char c = s[i];
            if (c >= '0' && c <= '9') {
                num = num * 10 + c - '0';
            } else if (c == '(') {
                int j = i, cnt = 0;
                for (; i < n; ++i) {
                    if (s[i] == '(') ++cnt;
                    if (s[i] == ')') --cnt;
                    if (cnt == 0) break;
                }
                num = calculate(s.substr(j + 1, i - j - 1));
            }
            if (c == '+' || c == '-' || c == '*' || c == '/' || i == n - 1) {
                switch (op) {
                    case '+': curRes += num; break;
                    case '-': curRes -= num; break;
                    case '*': curRes *= num; break;
                    case '/': curRes /= num; break;
                }
                if (c == '+' || c == '-' || i == n - 1) {
                    res += curRes;
                    curRes = 0;
                }
                op = c;
                num = 0;
            }
        }
        return res;
    }
```

Bézout's identity

z = x y 的最大公约数

两杯子（x y）装水问题： 存在a x + b y = z (a b是整数) 所以只需要看target是否为z的倍数



欧拉路径全覆盖

```
    void visit(String airport) {
        while(map.containsKey(airport) && !map.get(airport).isEmpty())
            visit(map.get(airport).poll());
        res.add(0, airport);
    }
```



图union find 684

一个乱序list可以分成k个cyclic list (用union find找小list，根据union次数能知道count(cyclic list))，让 子list re-order需要的swap次数是len(cyclic list)-1，让整个list re-order需要的swap次数是len(list) - count(cyclic list)

```
    private class UF {
        private int[] parents;
        UF(int n) {
            parents = new int[n];
            for (int i = 0; i < n; i++) {
                parents[i] = i;
            }
        }
        
        private int find(int i) {
            if (parents[i] == i) {
                return i;
            }
            parents[i] = find(parents[i]);
            return parents[i];
        }
        
        public void union(int i, int j) {
            int a = find(i);
            int b = find(j);
            if (a != b) {
                parents[a] = b;
            }
        }
    }
```

Minimum spanning tree

```Java
// 1584 哈密顿距离
// Kruskal 把所有边放进优先队列，如果两个图没联通，算上这条边
class Solution {
    public int minCostConnectPoints(int[][] points) {
        int n=points.length;
        PriorityQueue<int[]> pq=new PriorityQueue<>((a,b)->a[0]-b[0]);
        for(int i=0;i<n-1;++i)
            for(int j=i+1;j<n;++j){
                int dist=Math.abs(points[i][0]-points[j][0])+Math.abs(points[i][1]-points[j][1]);
                pq.offer(new int[]{dist,i,j});
            }
        int res=0;
        Union un=new Union(n);
        while(!un.United()&&!pq.isEmpty()){
            int[] edge=pq.poll();
            int p1=edge[1];
            int p2=edge[2];
            if(un.Unite(p1,p2))
                res+=edge[0];
        }
        return res;
    }
    private class Union{
        int components;
        int[] component;
        Union(){};
        Union(int n){
            components=n;
            component=new int[n+1];
            for(int i=0;i<=n;++i)
                component[i]=i;
        }
        int Find(int a){
            if(component[a]!=a)
                component[a]=Find(component[a]);
            return component[a];
        }
        boolean Unite(int a,int b){
            if(Find(a)==Find(b))
                return false;
            --components;
            component[Find(a)]=b;
            return true;
        }
        boolean United(){return components==1;}
    }
}

//Prim's 任选一点开始标记为visited，找他未visit的点的最小边，连接上，然后找他俩的最小边...
class Solution {
    public int minCostConnectPoints(int[][] points) {
        PriorityQueue<int[]> pq=new PriorityQueue<>((a,b)->a[0]-b[0]);  //min heap
        int res=0,connected=0,i=0,n=points.length;
        boolean[] visited=new boolean[n];
        while(++connected<n){
            visited[i]=true;
            for(int j=1;j<n;++j)
                if(!visited[j])
                    pq.offer(new int[]{Math.abs(points[i][0]-points[j][0])+Math.abs(points[i][1]-points[j][1]),j});
            while(visited[pq.peek()[1]])
                pq.poll();
            int[] curr=pq.poll();
            res+=curr[0];
            i=curr[1];
        }
        return res;
    }
}
```
