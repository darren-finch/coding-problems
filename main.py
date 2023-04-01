# # Permutations solution, had to look at it for this one. Still trying to understand.
# class Solution:
#     def permute(self, nums: List[int]) -> List[List[int]]:
#         result = []

#         if len(nums) == 1:
#             return [nums[:]]

#         for i in range(len(nums)):
#             n = nums.pop(0)
#             perms = self.permute(nums)
#             for perm in perms:
#                 perm.append(n)

#             nums.append(n)
#             result.extend(perms)

#         return result


# # Too sleepy to remove element
# class Solution:
#     def removeElement(self, nums: List[int], val: int) -> int:
#         p1 = 0
#         p2 = 0
#         k = 0

#         while p1 < len(nums) and p2 < len(nums):
#             if val == nums[p2]:
#                 nums[p1] = val
#                 val = nums[p2]
#                 p1 += 1
#                 k += 1
#             p2 += 1

#         return k


# // WIP
# class Solution:
#     def searchInsert(self, nums: List[int], target: int) -> int:
#         l = 0
#         r = len(nums)

#         while l < r:
#             mid = int((l + r) / 2)

#             if nums[mid] == target:
#                 return mid
#             elif nums[mid] < target:
#                 l = mid + 1
#             elif target < nums[mid]:
#                 r = mid - 1

#         return l


# // WIP: Binary search off the top of my head
# class Solution:
#     def findNum(nums: List[int], target: int):
#         l = 0
#         r = len(nums) - 1

#         while l < mid:
#             mid = r + l / 2

#             if target == mid:
#                 return target
#             elif mid < target:
#                 l = mid
#             elif target < mid:
#                 l = mid

#         return -1


# # Definition for a binary tree node.
# # class TreeNode(object):
# #     def __init__(self, val=0, left=None, right=None):
# #         self.val = val
# #         self.left = left
# #         self.right = right
# class Solution(object):
#     def binaryTreePaths(self, root):
#         """
#         :type root: TreeNode
#         :rtype: List[str]
#         """
#         if root == None:
#             return []

#         pathsList = []
#         rootIsLeaf = root.left == None and root.right == None
#         if rootIsLeaf:
#             pathsList = [str(root.val)]
#         else:
#             pathsList = self.binaryTreePaths(root.left) + self.binaryTreePaths(root.right)
#             for i in range(len(pathsList)):
#                 pathsList[i] = str(root.val) + "->" + pathsList[i]

#         return pathsList


# class Solution(object):
#     def threeSum(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: List[List[int]]
#         """
#         if len(nums) < 3:
#             return []


# # Definition for a binary tree node.
# # class TreeNode(object):
# #     def __init__(self, val=0, left=None, right=None):
# #         self.val = val
# #         self.left = left
# #         self.right = right
# class Solution(object):
#     def sortedArrayToBST(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: TreeNode
#         """
#         # leftPointer = largest value to left of middle, rightPointer = largest value to right of middle
#         middleIndex = (len(nums) - 1) / 2
#         middle = nums[middleIndex]
#         left = nums[middleIndex - 1]
#         right = nums[len(nums) - 1]


#     def traverseArray(self, nums, tree, left, right):
#         middleIndex = (left + right) / 2
#         middle = nums[middleIndex]
#         left = nums[middleIndex - 1]
#         right = nums[right - 1]


# class Solution(object):
#     solutions = []
#     potentialSolution = []
#     def change(self, amount, coins):
#         """
#         :type amount: int
#         :type coins: List[int]
#         :rtype: int
#         """
#         for coinAmt in coins:
#             balance = amount - coinAmt
#             if balance == 0:
#                 if potentialSolution not in solutions:
#                     solutions += potentialSolution
#             elif balance > 0:
#                 change(balance, coins)


#         return len(solutions)


# class Solution(object):

#     def change(self, amount, coins):
#         """
#         :type amount: int
#         :type coins: List[int]
#         :rtype: int
#         """
#         if amount == 0:
#             return 1
#         if len(coins) == 0:
#             return 0


# class Solution(object):
#     def countOdds(self, low, high):
#         """
#         :type low: int
#         :type high: int
#         :rtype: int
#         """
#         if low % 2 == 0 and high % 2 == 0:
#             return ((high - low + 1) / 2) - 1
#         elif low % 2 != 0 and high % 2 != 0:
#             return ((high - low + 1) / 2) + 1

#         return ((high - low + 1) / 2)


# def matrix_addition(a, b):
#     # your code here
#     n = len(a)
#     result = [[]]
#     for i in range(n):
#         for j in range(n):
#             result[i][j] = a[i][j] + b[i][j]

#     return result


# class Solution(object):
# def removeElement(self, nums, val):
#     """
#     :type nums: List[int]
#     :type val: int
#     :rtype: int
#     """
#     p1 = 0
#     p2 = 1
#     for num in nums:
#         if num != val:


# class Solution(object):
#     def longestCommonPrefix(self, strs):
#         """
#         :type strs: List[str]
#         :rtype: str
#         """
#         # loop thru arr
#         # Look at same char index for all elements in arr
#         # if char index out of range or current char is different than the ones in this round so far, break
#         # increase char index

#         ans = ""
#         charIndex = 0
#         stop = false

#         while(true):
#             curChar = strs[0][charIndex]
#             for (word in strs):
#                 if (charIndex > len(word) - 1 or word[charIndex] != curChar):
#                     stop = true
#                     break
#             if (stop)
#                 break
#             charIndex += 1
#             ans += curChar

#         return ans


# class Solution(object):
#     def longestCommonPrefix(self, strs):
#         """
#         :type strs: List[str]
#         :rtype: str
#         """
#         # loop thru arr
#         # Look at same char index for all elements in arr
#         # if char index out of range or current char is different than the ones in this round so far, break
#         # increase char index
#         if len(strs) < 1 or len(strs[0]) < 1:
#             return ""

#         ans = ""
#         charIndex = 0

#         while(charIndex < len(strs[0])):
#             curChar = strs[0][charIndex]
#             for word in strs:
#                 if charIndex > len(word) - 1 or word[charIndex] != curChar:
#                     return ans
#             charIndex += 1
#             ans += curChar

#         return ans


# class Solution(object):
#     def makesquare(self, matchsticks):
#         """
#         :type matchsticks: List[int]
#         :rtype: bool
#         """
#         # Algorithm:
#         # The goal is to divide the array into equal quarters
#         # Sum the array and divide that by 4. That is your length target. If this is a decimal, throw the thing out.
#         # When you get done with the algorithm, you want to see an array with 4 elements of equal length. I think.
#         return True


# class Solution(object):
#     def isPalindrome(self, x):
#         """
#         :type x: int
#         :rtype: bool
#         """
#         if x < 0:
#             return false

#         return str(x).reverse()
#     def reverse(self, s):
#         lenth = len(s)
#         charIndex = len(s) - 1
#         while charIndex > 0:
#             s[len() - i]

# def hello_world():
#     print("Hello world!")


import queue


class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        # Simple algorithm
        # Turn number into string
        # Reverse all of the string except for negative sign if it exists
        # Turn reversed string back into number

        # Algorithm for unsigned integer:
        # Start with empty string s
        # Until x is 0
        # Concat x mod 2 onto s
        # If len(s) > 32 then return 0
        # If loop completes, return s as an integer, which would just require summing up the powers of two

        # Algorithm for signed integer
        # Turn integer into unsigned integer and store whether it was negative
        # Do algorithm for unsigned integer
        # Return


class Solution(object):
    def twoSum(self, nums, target):
        seenNumbers = {}
        for i, num in enumerate(nums):
            neededDiff = target - num
            if not seenNumbers.has_key(neededDiff):
                seenNumbers[num] = i
            else:
                return [seenNumbers[neededDiff], i]
        return [-1, -1]


def disemvowel(string):
    vowels = 'aeiouAEIOU'
    new_string = ''
    for i in string:
        if i not in vowels:
            new_string += i
    return new_string


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if root == None:
            return []
        return [root.val] + self.preorderTraversal(root.left) + self.preorderTraversal(root.right)


class Solution:
    def isHappy(self, n: int) -> bool:
        lastResultAsStr = str(n)
        result = 0

        while result != 1:
            result = 0
            for char in lastResultAsStr:
                result += pow(int(char), 2)
            resultAsStr = str(result)
            if resultAsStr == lastResultAsStr:
                break
            lastResultAsStr = resultAsStr

        return result == 1


class Solution:
    def answerQueries(self, nums: List[int], queries: List[int]) -> List[int]:
        n = len(nums)
        m = len(queries)

        sumOfNums = 0
        for num in nums:
            sumOfNums += num

        answer = []
        for query in queries:
            sumOfDeletedElementsForQuery = 0
            deletedElements = {}

            while query < sumOfNums - sumOfDeletedElementsForQuery:
                smallestElementToDelete = nums[0]
                for num in nums:
                    if num < smallestElementToDelete and num not in deletedElements:
                        smallestElementToDelete == num

                deletedElements[smallestElementToDelete] = smallestElementToDelete
                sumOfDeletedElementsForQuery += smallestElementToDelete

            answer.append(n - len(deletedElements))

        return answer


class Solution:
    def canJump(self, nums: List[int]) -> bool:
        reachableIndices = [False for i in range(len(nums))]
        for j in range(len(nums)):
            num = nums[j]

            if num == 0 and j < len(nums) - 1 and reachableIndices[j + 1] == False:
                return False

            for k in range(j, j + num + 1):
                if k < len(reachableIndices):
                    reachableIndices[k] = True

        return True


def make_readable(seconds):
    hr = int(seconds / 3600)
    min = int((seconds - (hr * 3600)) / 60)
    sec = int(seconds - (hr * 3600) - (min * 60))

    hrStr = str(hr) if hr > 9 else f'0{hr}'
    minStr = str(min) if min > 9 else f'0{min}'
    secStr = str(sec) if sec > 9 else f'0{sec}'

    return f'{hrStr}:{minStr}:{secStr}'


def move_zeros(list):
    return [num for num in list if num != 0] + [num for num in list if num == 0]


def zeros(n):
    return int(n / 5)


def is_valid_walk(walk):
    if len(walk) != 10:
        return False

    verticalDirections = {
        'n': 1,
        's': -1
    }
    horizontalDirections = {
        'w': 1,
        'e': -1
    }
    verticalDelta = 0
    horizontalDelta = 0
    for direction in walk:
        if direction in verticalDirections:
            verticalDelta += verticalDirections[direction]
        elif direction in horizontalDirections:
            horizontalDelta += horizontalDirections[direction]

    return verticalDelta == 0 and horizontalDelta == 0


def find_it(seq):
    seenNums = {}
    for num in seq:
        occurrencesCnt = seenNums[num] if num in seenNums else 0
        seenNums[num] = occurrencesCnt + 1

    for seenNum in seenNums.keys():
        occurrencesCnt = seenNums[seenNum]
        if occurrencesCnt % 2 != 0:
            return seenNum

    return -1


def dirReduc(arr):
    opposites = {
        ("NORTH", "SOUTH"),
        ("SOUTH", "NORTH"),
        ("EAST", "WEST"),
        ("WEST", "EAST")
    }

    while True:
        p1 = 0
        p2 = 1
        lastLen = len(arr)
        while p2 < len(arr):
            if (arr[p1], arr[p2]) in opposites:
                arr.remove(arr[p1])
                arr.remove(arr[p2 - 1])
            else:
                p1 += 1
                p2 += 1

        if len(arr) == lastLen:
            break

    return arr


def pig_it(text):
    result = ''
    firstLetterOfNextWord = ''
    for i in range(len(text)):
        char = text[i]

        # With this implementation we assume no leading punctuation or whitespace.
        if i == 0:
            firstLetterOfWord = char
        else:
            lastChar = text[i - 1]
            foundNewWord = char.isalpha() and not lastChar.isalpha()
            foundEndOfWordBeforeEndOfText = (
                not char.isalpha() and lastChar.isalpha())
            foundEndOfWordAtEndOfText = (i == len(text) - 1 and char.isalpha())

            if foundNewWord:
                firstLetterOfWord = char
                # This case will occur when a single letter appears at the end of the string.
                if foundEndOfWordAtEndOfText:
                    result = ''.join([result, firstLetterOfWord + 'ay' + char])
            elif foundEndOfWordBeforeEndOfText:
                result = ''.join([result, firstLetterOfWord + 'ay' + char])
            elif foundEndOfWordAtEndOfText:
                result = ''.join([result, char + firstLetterOfWord + 'ay'])
            else:  # just a normal letter
                result = ''.join([result, char])

    return result


def parse_int(string):
    return  # number


def pick_peaks(arr):
    if len(arr) < 3:
        return {[], []}

    pos = []
    peaks = []
    curPotentialMaximaIndex = i
    for i in range(len(arr)):
        if i == 0 or i == len(arr) - 1:
            if element < arr[curPotentialMaximaIndex]:
                pos.append(curPotentialMaximaIndex)
                peaks.append(arr[curPotentialMaximaIndex])
            continue

        element = arr[i]
        if element > arr[curPotentialMaximaIndex]:
            curPotentialMaximaIndex = i
        elif element < arr[curPotentialMaximaIndex]:
            pos.append(curPotentialMaximaIndex)
            peaks.append(arr[curPotentialMaximaIndex])

    return {pos, peaks}


def dirReduc(arr):
    opposites = {
        ("EAST", "WEST"),
        ("NORTH", "SOUTH"),
        ("WEST", "EAST"),
        ("SOUTH", "NORTH"),
    }

    p1 = 0
    p2 = 1
    while p2 < len(arr):
        if (arr[0], arr[1]) in opposites:
            arr = arr[:p1] + arr[p1] + arr[p1 + 1:len(arr)]
            arr.remove(arr.index(p2))

            if p2 > 1:
                p1 -= 1
                p2 -= 1
        else:
            p1 += 1
            p2 += 1


def same_structure_as(original, other):
    print(f'original: {original}\nother: {other}')

    if len(original) != len(other):
        console.log('Is this reached')
        return False

    if (original is not list and other is not list):
        return True

    for i in range(len(original)):
        originalElement = original[i]
        otherElement = other[i]
        if not same_structure_as(originalElement, otherElement):
            return False

    return True


class Interpreter:
    def execute(self, programInstructionsAsStr):
        for instructionAsStr in programInstructions:
            instruction = Instruction


instructionToClassMapping = {
    'mov': MoveInstruction,
    # WIP
}


class Instruction:
    def __init__(self, string):
        self.string = string


class MoveInstruction(Instruction):
    def __init__(self, string):
        super.__init__(string)
        self.arguments = string


def getProgramInstructionFromStr(string):
    if string.startswith('mov'):
        return MoveInstruction(string)


def simple_assembler(program):
    # return a dictionary with the registers
    return {}


def next_smaller(n):
    # Strategy
    # Potentiall bubble the smaller numbers towards the top
    pass


def to_underscore(string):
    if type(string) == int or type(string) == float:
        return str(string)

    result = ''
    for i in range(len(string)):
        char = string[i]
        if char.isupper():
            if i != 0:
                result = ''.join([result, '_'])
            result = ''.join([result, char.lower()])
        else:
            result = ''.join([result, char])

    return result


def loop_size(node):
    curNode = node
    seenNodes = {}
    distanceFromStart = 0
    while curNode != None:
        if curNode in seenNodes:
            return distanceFromStart - seenNodes[curNode]
        else:
            seenNodes[curNode] = distanceFromStart
            distanceFromStart += 1
        curNode = curNode.next

    return 0


def exp_sum(n):
    if n == 1:
        return n

    addend = n


# The retry of exp_sum was still not successful whatsoever lol.
def exp_sum(n):
    pass


def justify(text, width):
    # Iterate through the characters
    # When you pass enough characters that you go past the width, stop
    # Once stopped, either try to iterate backward until you find a space.
    # If instead of a space, you find the beginning of the array, that means you'll just have to break the word up (this case won't occur for this problem but it may in the future).
    # Once you find where to break the sentence, start adding all the words up to the line break, and space them out how they need to be
    # (this will be calculated from the number of letters crossed vs the number of spaces crossed when determining where to put the line break.)
    pass


def justify(text, width):
    # Input: abc abcde ab ab abcdefg
    # Justify to 15 chars
    # Output: abc abcde ab abcdefg

    numOfCharsSeen = 0
    numOfSpaces = 0
    words = []

    prevChar = None
    for char in text:
        numOfCharsSeen += 1
        if char == ' ':
            numOfSpaces += 1

#         if prevChar == None or (prevChar == ' ' and char != ' '):


def find_outlier(integers):
    evensCount = 0
    evensSum = 0
    oddsSum = 0

    for num in integers:
        if num % 2 == 0:
            evensSum += num
            evensCount += 1
        else:
            oddsSum += num

    if evensCount == 1:
        return evensSum
    else:
        return oddsSum


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root == None:
            return []

        ans = [[]]
        curDepth = 0
        queue = [(root, curDepth)]

        while len(queue) > 0:
            nextNodeDepthPair = queue.pop(0)
            nextNode = nextNodeDepthPair[0]
            nextNodeDepth = nextNodeDepthPair[1]

            if nextNode.left:
                queue.append((nextNode.left, nextNodeDepth + 1))
            if nextNode.right:
                queue.append((nextNode.right, nextNodeDepth + 1))

            if curDepth == nextNodeDepth:
                ans[curDepth].append(nextNode.val)
            else:
                ans.append([nextNode.val])
                print(curDepth)
                curDepth = nextNodeDepth

        return ans

    class Solution:
        def merge(self, intervals: List[List[int]]) -> List[List[int]]:
            # Iterate over each item
            #   For each item, iterate over each item again, exluding the current item from the first loop
            #       For each item pair, check if the first item overlaps with the second
            #       If they do, set the current merged pair = [min start of the two pairs, max finish of the two pairs]
            #       If they don't, move on
            #   Add current merged pair to output
            # Return output
            pass


class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        # Probably use bFS
        pass


class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        hammingDistance = 0

        while x > 0 or y > 0:
            if (x % 2 == 0 and y % 2 == 1) or (x % 2 == 1 and y % 2 == 0):
                hammingDistance += 1
            x = int(x / 2)
            y = int(y / 2)

        return hammingDistance


class Solution:
    def findBall(self, grid: List[List[int]]) -> List[int]:
        colsWithDroppedBall = [-1 for col in range(len(grid[0]))]

        for startingCol in range(len(grid[0])):
            curRow = 0
            curCol = startingCol
            ballStopped = False
            while curRow < len(grid):
                curElement = grid[curRow][curCol]
                ballWillGoTowardsWall = (
                    (curCol == 0 and curElement == -1) or (curCol == len(grid[0]) and curElement == 1))

                if ballWillGoTowardsWall:
                    ballStopped = True
                    break

                elementToLeft = grid[curRow][curCol -
                                             1] if curCol > 0 else None
                elementToRight = grid[curRow][curCol +
                                              1] if curCol < len(grid[0]) - 1 else None
                ballWillGoTowardsV = ((curElement == 1 and (elementToRight != 1 if elementToRight != None else True))
                                      or (curElement == -1 and (elementToLeft != -1 if elementToLeft != None else True)))

                if ballWillGoTowardsV:
                    ballStopped = True
                    break

                curRow += 1
                curCol += curElement

            if not ballStopped:
                colsWithDroppedBall[startingCol] = curCol
        return colsWithDroppedBall


class Solution:
    def isHappy(self, n: int) -> bool:
        # Could not come up with solution in 15 mins but I tried
        pass


class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        # candidates = [3,2,7,6], target = 7
        pass


class Solution:
    def isHappy(self, n: int) -> bool:
        if n == 1:
            return True

        seenNumbers = {}

        while n not in seenNumbers and n != 1:
            seenNumbers[n] = n
            n = self._getSumOfSquaresOfDigits(n)

        return n == 1

    def _getSumOfSquaresOfDigits(self, n: int) -> int:
        ans = 0
        nAsStr = str(n)
        for digit in nAsStr:
            ans += pow(int(digit), 2)
        return ans

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next


class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        prev = None
        curr = head
        currIndex = 0
        next = head.next if head else None
        firstNodeOfReversedSectionAfterReverse = None
        lastNodeOfReversedSectionAfterReverse = None
        nodeBeforeReversedSection = None
        nodeAfterReversedSection = None

        while currIndex <= right:
            if left <= currIndex:
                # If currIndex equals left, we don't need to link to prev node
                if currIndex == left:
                    lastNodeOfReversedSectionAfterReverse = curr
                    nodeBeforeReversedSection = prev
                    prev = None

                if currIndex == right:
                    firstNodeOfReversedSectionAfterReverse = curr
                    nodeAfterReversedSection = next

                # Reverse
                curr.next = prev

            # Move pointers
            prev = curr
            curr = next
            currIndex += 1
            next = next.next if next else None

        lastNodeOfReversedSectionAfterReverse.next = nodeAfterReversedSection
        if nodeBeforeReversedSection == None:
            return firstNodeOfReversedSectionAfterReverse
        else:
            nodeBeforeReversedSection.next = firstNodeOfReversedSectionAfterReverse
            return head


class Solution:
    # [-5, 5, 5, -6]
    def maxAbsoluteSum(self, nums: list[int]) -> int:
        pass


class Solution:
    def jump(self, nums: List[int]) -> int:
        # Return minimum number of jumps to reach end starting at startPos
        def getMinJumpsToReachEndFromStartPos(startPos: int) -> int:
            if len(nums) - 1 < startPos or nums[startPos] == 0:
                return 1000000

            if len(nums) - 1 == startPos:
                return 0

            maxPossibleJumpLen = min(nums[startPos], len(nums) - 1 - startPos)

            minPossibleJumps = 1000000
            for curJumpLen in range(maxPossibleJumpLen + 1, 1, -1):
                minPossibleJumpsForCurJumpLen = 1 + \
                    getMinJumpsToReachEndFromStartPos(startPos + curJumpLen)
                if minPossibleJumpsForCurJumpLen < minPossibleJumps:
                    minPossibleJumps = minPossibleJumpsForCurJumpLen

            return minPossibleJumps
        return getMinJumpsToReachEndFromStartPos(0)


class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return 1

        wiggleIntervals = 0
        lastDiff = None
        i = 0

        while i < len(nums) - 1:
            curDiff = nums[i + 1] - nums[i]

            if (i == 0 and curDiff != 0) or (lastDiff != None and ((curDiff > 0 and lastDiff < 0) or (curDiff < 0 and lastDiff > 0))):
                lastDiff = curDiff
                wiggleIntervals += 1

            i += 1

        return wiggleIntervals + 1


class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        # I really tried. WIP
        pass


# Had to look at solution for this one, got stuck
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        # cbacdcbcabcd

        lastOccurrences = {}
        stack = []
        visited = set()

        for i in range(len(s)):
            lastOccurrences[s[i]] = i

        for i in range(len(s)):
            if s[i] not in visited:
                while 0 < len(stack) and s[i] < stack[-1] and i < lastOccurrences[stack[-1]]:
                    visited.remove(stack.pop())

                stack.append(s[i])
                visited.add(s[i])

        return ''.join(stack)


class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        p1 = 0
        p2 = 1
        minLen = len(nums)
        minLenWasSet = False
        curSum = nums[0]

        while p2 < len(nums):
            curSum += nums[p2]

            while target <= curSum - nums[p1] and p1 < p2:
                curSum -= nums[p1]
                p1 += 1

            potentialNextMinLen = p2 - p1 + 1
            if target <= curSum and potentialNextMinLen <= minLen:
                minLen = potentialNextMinLen
                minLenWasSet = True

            p2 += 1

        return minLen if minLenWasSet else 0

# Below is the interface for Iterator, which is already defined for you.
#
# class Iterator:
#     def __init__(self, nums):
#         """
#         Initializes an iterator object to the beginning of a list.
#         :type nums: List[int]
#         """
#
#     def hasNext(self):
#         """
#         Returns true if the iteration has more elements.
#         :rtype: bool
#         """
#
#     def next(self):
#         """
#         Returns the next element in the iteration.
#         :rtype: int
#         """

# nums = []
# iter.next = None
# next = None


class PeekingIterator:
    def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self.iterator = iterator
        self.next = None

    def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        if self.next == None and self.iterator.hasNext():
            self.next = self.iterator.next()

        print("call to peek with ret val = " + self.next)

        return self.next

    def next(self):
        """
        :rtype: int
        """
        prevNext = self.next

        if self.next == None and self.iterator.hasNext():
            self.next = self.iterator.next()
            prevNext = self.next

        if self.iterator.hasNext():
            self.next = self.iterator.next()

        print("call to next with ret val = " + prevNext)
        return prevNext

    def hasNext(self):
        """
        :rtype: bool
        """
        return self.next != None or self.iterator.hasNext()


# Your PeekingIterator object will be instantiated and called as such:
# iter = PeekingIterator(Iterator(nums))
# while iter.hasNext():
#     val = iter.peek()   # Get the next element but not advance the iterator.
#     iter.next()         # Should return the same value as [val].

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
        if head == None:
            return None

        listAsArr = self.sortedListToArr(head)
        return self.sortedArrToBST(listAsArr, 0, len(listAsArr) - 1)

    def sortedListToArr(self, head: Optional[ListNode]):
        arr = []

        curr = head
        while curr:
            arr.append(curr.val)
            curr = curr.next

        return arr

    def sortedArrToBST(self, arr, left, right):
        if right < left:
            return None
        if left == right:
            return TreeNode(arr[left], None, None)

        mid = (left + right) // 2
        return TreeNode(arr[mid], self.sortedArrToBST(arr, left, mid - 1), self.sortedArrToBST(arr, mid + 1, right))


class Solution:
    def trap(self, height: List[int]) -> int:
        stack = []
        ans = 0

        for i in range(len(height)):
            while 0 < len(stack) and height[stack[-1]] < height[i]:
                poppedHeightIndex = stack.pop()

                if len(stack) == 0:
                    break

                # THIS IS WHERE ALGORITHM NEEDS FIXING.
                heightDiff = height[i] - \
                    (height[stack[-1]] - height[poppedHeightIndex])
                indexDiff = i - stack[-1]

                ans += heightDiff * indexDiff
            stack.append(i)

        return ans


class Solution:
    def trap(self, height: List[int]) -> int:
        stack = []
        totalRainwaterVolume = 0

        for i in range(len(height)):
            while 0 < len(stack) and height[stack[-1]] < height[i]:
                poppedElem = stack.pop()

                # At the base of a hill with no previously greater height behind us, nothing needs to be done.
                if len(stack) == 0:
                    break

                # We know we just popped off an element of the stack that was less than the current element.
                # We also know that because the stack is a monotonically decreasing stack, that the previous element must be greater than the popped element.
                # Therefore, we know that rainwater could be stored between the current element and the new last element of the stack.
                #
                # The important thing to understand is that we calculate rainwater moving from the bottom right to the top left.
                # Therefore, at this point, all rainwater at heights <= height[poppedElem] between stack[-1] and poppedElem has been calculated.
                # Therefore, to get the height of the rainwater container we are currently looking at,
                # we need to take the minimum of either stack[-1] or height[i] and subtract the height of the popped element.
                heightDiff = min(height[stack[-1]],
                                 height[i]) - height[poppedElem]

                indexDiff = i - (stack[-1] + 1)

                rainwaterVolume = heightDiff * indexDiff
                totalRainwaterVolume += rainwaterVolume

            stack.append(i)

        return totalRainwaterVolume


class Solution:
    def decodeString(self, s: str) -> str:
        stack = []
        output = ''
        temp = ''

        for char in s:
            if char.isdigit():
                stack.append([int(char), ''])
            elif char == ']':
                lastElem = stack.pop()
                temp = ''.join([lastElem[1]
                               for i in range(lastElem[0])]) + temp
                if len(stack) == 0:
                    output += temp
            elif char != '[':
                print(stack)
                stack[-1][1] += char
        return output


class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        # Solution is:
        # O(n) time complexity
        # O(3n) space complexity
        # Inspired by Dijkstra

        infinity = 1000

        queue = Queue()
        queue.put((0, 0))
        visited = set()

        # for i in range(len(grid)):
        #     for j in range(len(grid[i])):
        #         unvisited.add((i, j))

        minPathSums = [[1000 for j in range(len(grid[i]))]
                       for i in range(len(grid))]
        minPathSums[0][0] = grid[0][0]

        while not queue.empty():
            curElemRow, curElemCol = queue.get()
            curElemValue = grid[curElemRow][curElemCol]

            # Check if path sum from this element will be less than the path sums
            # found by reaching neighbor a different way.
            if minPathSums[curElemRow + 1][curElemCol]

        lastRowIndex = len(minPathSums) - 1
        lastColumnIndex = len(minPathSums[lastRowIndex]) - 1
        return minPathSums[lastRowIndex][lastColumnIndex]


class Solution:
    def divisorSubstrings(self, num: int, k: int) -> int:
        kbeauty = 0
        start = 0
        end = k
        numAsStr = str(num)

        while end <= len(numAsStr):
            numFromSlidingWindow = int(numAsStr[start:end])
            if numFromSlidingWindow != 0 and num % numFromSlidingWindow == 0:
                kbeauty += 1
            start += 1
            end += 1

        return kbeauty
