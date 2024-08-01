import collections

# p1110
in_S = collections.Counter()
out_S = collections.Counter()
in_count = 0
out_count = 0
for x in l:
    if x in S:
        in_S[x] += 1
        in_count += 1
    else:
        out_S[x] += 1
        out_count += 1
in_max = max(in_S.values())
out_max = max(out_S.values())

res_in = out_count + (in_count - in_max) * 2
res_out = in_count + (out_count - out_max) * 2
print(min(res_in, res_out))


# p1111
nm = input()
N, M = list(map(int, nm.split(' ')))
downstream = []
for _ in range(M):
    ab = input()
    a, b = list(map(int, ab.split(' ')))
    downstream.append(a, b)

indegree = [0 for _ in range(N)]
outdegree = [0 for _ in range(N)]

for a, b in downstream:
    indegree[b] += 1
    outdegree[a] += 1

n_end = sum(x==0 for x in outdegree)

if n_end == 1:
    for i, x in enumerate(outdegree):
        if x == 0:
            print(i)
else:
    m = 0
    ans = -1
    for i, x in enumerate(indegree):
        if x >= m:
            ans = i
            m = x
    print(ans)


# p1112
N, M, neighbors

edges = collections.defaultdict(list)
for a, b in neighbors:
    edges[a].append(b)
    edges[b].append(a)

visited = set()
def dfs(i):
    visited.add(i)
    g = 1
    for j in edges[i]:
        if j not in visited:
            g += dfs(j)
    return g

numbers = []
for i in range(1, N):
    if i not in visited:
        n = dfs(i)
        numbers.append(n)

# https://interviewing.io/questions/partition-to-k-equal-sum-subsets
def canPartitionKSubsets(nums: List[int], k: int) -> bool:
    total_sum = sum(nums)

    # Return False if the total sum could not be split into K equal subsets
    if total_sum % k != 0:
        return False
    
    # The target the sum for each subset
    target = total_sum // k
    # Keep track of the elements that have been taken
    taken = ['0'] * len(nums)
    
    # memorize the solutions for the subproblems to avoid duplicate calculations
    memo = {}

    # input parameters
    # - number of subsets we have formed
    # - the current sum for the current subset we are working on
    # - the index of the element in the array we need to try on
    def backtrack(num_completed: int, current_sum: int, index: int) -> bool:
        # The current subproblem status
        taken_str = ''.join(taken)
        
        # Return True if we have successfully formed k-1 subsets
        # as the last subset will for sure can be formed
        if num_completed == k-1:
            return True
        
        if current_sum > target:
            return False
        
        # No need to recalculate if the subproblem has been calculated before
        if taken_str in memo:
            return memo[taken_str]
        
        # Successfully formed one subset, try to form a new one
        if current_sum == target:
            memo[taken_str] = backtrack(num_completed + 1, 0, 0)
            return memo[taken_str]
        
        # Try out each possibility to group into the current subset
        for i in range(index, len(nums)):
            # try nums[i] if it is not taken
            if taken[i] == '0':
                taken[i] = '1'
                # we only need to try out i+1, i+2 elements and onwards
                # as the previous elements has already been tried out in previous bracktrack loops
                if backtrack(num_completed, current_sum + nums[i], i+1):
                    return True
                # undo if nums[i] is not possible
                taken[i] = '0'
            # Early break. Every element should be grouped into a subset. 
            # If this element cannot be grouped into a subset, then this problem is not solvable
            if taken == ['0'] * len(nums):
                break
        
        memo[taken_str] = False
        return memo[taken_str]
        
    return backtrack(0, 0, 0)

ans = []
for k in range(1, len(numbers)+1):
    if canPartitionKSubsets(numbers, k):
        ans.append(k)
print(*ans)


# p1113
M, N

dp = [[0 for j in range(N+1)] for i in range(M+1)]

for c, m, v in zip(core, mem, value):
    for i in range(M, c-1, -1):
        for j in range(N, m-1, -1):
            x = i - c
            y = j - m
            dp[i][j] = max(dp[i][j], dp[x][y] + v)

print(dp[-1][-1])


# p1115
import heapq

budget, source, target, N, M
routes

A = collections.defaultdict(dict)
for a, b, delay, fee in routes:
    A[a][b] = [delay, fee]

cheapest = {i:float('inf') for i in range(1, N+1)}
cheapest[source] = 1

ans_delay = -1
ans_fee = -1
heap = [[0, 0, source]]
while heap:
    total_delay, total_fee, i = heapq.heappop(heap)
    if total_fee >= cheapest[i]:
        continue
    if i == target:
        ans_delay = total_delay
        ans_fee = total_fee
        break
    cheapest[i] = total_fee
    for j, (delay, fee) in A[i].items():
        if total_fee + fee <= budget:
            heapq.heappush(heap, [total_delay+delay, total_fee+fee, j])

print(ans_delay)
print(ans_fee)


# p1116
M, N, ads<income, cost, limit>

dp = [0 for _ in range(M+1)]
for income, cost, limit in ads:
    for _ in range(limit):
        for i in range(M, cost-1, -1):
            dp[i] = max(dp[i], dp[i-cost]+income)
print(dp[-1])


# p1117
S1, S2, table

def lcs(w1, w2):
    dp = [[0 for j in range(len(w2)+1)] for i in range(len(w1)+1)]
    for i in range(1, len(dp)):
        for j in range(1, len(dp[0])):
            match = w1[i-1] == w2[j-1]
            dp[i][j] = max(dp[i-1][j-1]+match, dp[i][j-1], dp[i-1][j])
    return dp[-1][-1]

w2i = {}
for i, words in enumerate(table):
    for word in words:
        w2i[word] = i

similar = True
for w1, w2 in zip(S1, S2):
    if w2i[w1] != w2i[w2]:
        similar = False
        break
ans = 0
if similar:
    print(ans)
else:
    for w1, w2 in zip(S1, S2):
        ans += lcs(w1, w2)
    print(ans)
