# Permutations solution, had to look at it for this one. Still trying to understand.
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        result = []
        
        if len(nums) == 1:
            return [nums[:]]
        
        for i in range(len(nums)):
            n = nums.pop(0)
            perms = self.permute(nums)
            for perm in perms:
                perm.append(n)
            
            nums.append(n)
            result.extend(perms)
            
        return result



# Too sleepy to remove element
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        p1 = 0
        p2 = 0
        k = 0
        
        while p1 < len(nums) and p2 < len(nums):
            if val == nums[p2]:
                nums[p1] = val
                val = nums[p2]
                p1 += 1
                k += 1
            p2 += 1
                
        return k



// WIP
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        l = 0
        r = len(nums)
        
        while l < r:
            mid = int((l + r) / 2)
            
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                l = mid + 1
            elif target < nums[mid]:
                r = mid - 1
        
        return l



// WIP: Binary search off the top of my head
class Solution:
    def findNum(nums: List[int], target: int):
        l = 0
        r = len(nums) - 1
        
        while l < mid:
            mid = r + l / 2

            if target == mid:
                return target
            elif mid < target:
                l = mid
            elif target < mid:
                l = mid

        return -1



# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        if root == None:
            return []
        
        pathsList = []
        rootIsLeaf = root.left == None and root.right == None
        if rootIsLeaf:
            pathsList = [str(root.val)]
        else:
            pathsList = self.binaryTreePaths(root.left) + self.binaryTreePaths(root.right)
            for i in range(len(pathsList)):
                pathsList[i] = str(root.val) + "->" + pathsList[i]
                
        return pathsList



class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if len(nums) < 3:
            return []



# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        # leftPointer = largest value to left of middle, rightPointer = largest value to right of middle
        middleIndex = (len(nums) - 1) / 2
        middle = nums[middleIndex]
        left = nums[middleIndex - 1]
        right = nums[len(nums) - 1]

    
    def traverseArray(self, nums, tree, left, right):
        middleIndex = (left + right) / 2
        middle = nums[middleIndex]
        left = nums[middleIndex - 1]
        right = nums[right - 1]




class Solution(object):
    solutions = []
    potentialSolution = []
    def change(self, amount, coins):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        """
        for coinAmt in coins:
            balance = amount - coinAmt
            if balance == 0:
                if potentialSolution not in solutions:
                    solutions += potentialSolution
            elif balance > 0:
                change(balance, coins)
            
            
        return len(solutions)



class Solution(object):
    
    def change(self, amount, coins):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        """
        if amount == 0:
            return 1
        if len(coins) == 0:
            return 0



class Solution(object):
    def countOdds(self, low, high):
        """
        :type low: int
        :type high: int
        :rtype: int
        """
        if low % 2 == 0 and high % 2 == 0:
            return ((high - low + 1) / 2) - 1
        elif low % 2 != 0 and high % 2 != 0:
            return ((high - low + 1) / 2) + 1
        
        return ((high - low + 1) / 2)



def matrix_addition(a, b):
    # your code here
    n = len(a)
    result = [[]]
    for i in range(n):
        for j in range(n):
            result[i][j] = a[i][j] + b[i][j]
            
    return result



class Solution(object):
def removeElement(self, nums, val):
    """
    :type nums: List[int]
    :type val: int
    :rtype: int
    """
    p1 = 0
    p2 = 1
    for num in nums:
        if num != val:



class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        # loop thru arr
        # Look at same char index for all elements in arr
        # if char index out of range or current char is different than the ones in this round so far, break
        # increase char index
        
        ans = ""
        charIndex = 0
        stop = false
        
        while(true):
            curChar = strs[0][charIndex]
            for (word in strs):
                if (charIndex > len(word) - 1 or word[charIndex] != curChar):
                    stop = true
                    break
            if (stop)
                break
            charIndex += 1
            ans += curChar
        
        return ans



class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        # loop thru arr
        # Look at same char index for all elements in arr
        # if char index out of range or current char is different than the ones in this round so far, break
        # increase char index
        if len(strs) < 1 or len(strs[0]) < 1:
            return ""
        
        ans = ""
        charIndex = 0
        
        while(charIndex < len(strs[0])):
            curChar = strs[0][charIndex]
            for word in strs:
                if charIndex > len(word) - 1 or word[charIndex] != curChar:
                    return ans
            charIndex += 1
            ans += curChar
        
        return ans



class Solution(object):
    def makesquare(self, matchsticks):
        """
        :type matchsticks: List[int]
        :rtype: bool
        """
        # Algorithm:
        # The goal is to divide the array into equal quarters
        # Sum the array and divide that by 4. That is your length target. If this is a decimal, throw the thing out.
        # When you get done with the algorithm, you want to see an array with 4 elements of equal length. I think.
        return True



class Solution(object):
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if x < 0:
            return false
        
        return str(x).reverse()
    def reverse(self, s):
        lenth = len(s)
        charIndex = len(s) - 1
        while charIndex > 0:
            s[len() - i]