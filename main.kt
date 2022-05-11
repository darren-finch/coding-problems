fun twoSum(nums: IntArray, target: Int): IntArray {
    val seenNumbers = HashMap<Int, Int>()
    for (i in 0..nums.lastIndex) {
        val curVal = nums[i]
        val difference = target - curVal
        val otherValIndex = seenNumbers[difference]
        
        if (otherValIndex != null) {
            return intArrayOf(i, otherValIndex)
        } else {
            seenNumbers[curVal] = i
        }
    }
    
    return intArrayOf(); //wont be reached
}


`~
class Solution {
    fun addTwoNumbers(l1: ListNode?, l2: ListNode?): ListNode? {
        var p1 = l1
        var p2 = l2
        var carry = 0
        var ans: ListNode? = ListNode(0)
        var p3 = ans
        
        while (p1 != null || p2 != null || carry != 0) {
            val sum = ((p1?.`val` ?: 0) + (p2?.`val` ?: 0) + carry)
            p3?.`val` = sum % 10
            carry = sum / 10
            
            p1 = p1?.next
            p2 = p2?.next
            
            if (p1 != null || p2 != null || carry != 0)            
                p3?.next = ListNode(0)
            p3 = p3?.next
        }
        
        return ans
    }
}



class Solution {
    fun maxSubArray(nums: IntArray): Int {
        if (nums.size < 2) {
            return nums[0]
        }
        
        var curSum = nums[0]
        var maxSum = nums[0]
        
        for (i in 1..nums.lastIndex) {
            curSum += nums[i]
            if (curSum < 0) {
                curSum = nums[i]
                continue
            }
            if (curSum > maxSum) {
                maxSum = curSum
            }
        }
        
        return maxSum
    }
}



class Solution {
    fun isPalindrome(x: Int): Boolean {
        return x.toString().reversed() == x.toString()
    }
}



class Solution {
    fun longestCommonPrefix(strs: Array<String>): String {
        // Idk, and I really need sleep
        val prefixes = stringArrayOf("")
    }
}


class Solution {
    fun hasCycle(head: ListNode?): Boolean {
        var p1 = head
        var p2 = head.next
    }
}



object Solution {
    fun areaOrPerimeter(l: Int, w: Int) = if (l == w) l * w else (2 * l) + (2 * w);
}

class Solution {
    fun lengthOfLastWord(s: String): Int {
        var foundNonSpaceChar = false
        var length = 0;
        for (i in s.lastIndex downTo 0) {
            val curChar = s[i];
            
            if (curChar == ' ' && length > 0) {
                return length;
            }
            if (curChar != ' ') {
                length++;
            }
        }
        
        return length;
    }
}



class Solution {
    fun mergeTwoLists(list1: ListNode?, list2: ListNode?): ListNode? {
        var p1 = list1
        var p2 = list2
        var p1Next = list1!!.next
        var p2Next = list2!!.next
        var head = list1
        
        while (p1 != null) {
            if (p1Next == null) {
                p1!!.next = p2
                return head
            }
            if (p1Next!!.`val` >= p2!!.`val`) {
                p1!!.next = p2
                p2!!.next = p1Next
                p2 = p2Next
                p2Next = p2Next!!.next
            }
            
            p1 = p1!!.next
            p1Next = p1Next!!.next
        }
        
        return head
    }
}



class Solution {
    fun inorderTraversal(root: TreeNode?): List<Int> {
        val ans = listOf<Int>()
        
    }
    
    fun traverse(root: TreeNode?, ans: List<Int>) {
        while (root != null) {
            
        }
    }
}



class Solution {
    fun romanToInt(s: String): Int {
        // Create var sum
        // Loop through string
        // Add integer equivalent of { I, V, X, L, C, D, M } to sum
        // Detect { IV, IX, XL, XC, CD, CM} and add integer equivalent of those to sum
        // Detect by making sure index isn't at strength.length
        var sum = 0
        for (i in s.indices) {
            
        }
    }
    
    fun isNormalCharacter(character: Char): Boolean {
        return when (character) {
            is "I" -> true
            is "V" -> true
            is "X" -> true
            is "L" -> true
            is "C" -> true
            is "D" -> true
            is "M" -> true
        }
    }
}



class Solution {
    fun search(nums: IntArray, target: Int): Int {
        var left = 0
        var right = nums.lastIndex
        var mid = (left + right) / 2
        
        while (target != nums[mid] && left < right) {
            if (target > nums[mid]) {
                left = mid + 1
            } else {
                right = mid - 1
            }
            
            mid = (left + right) / 2
        }
        
        if (target != nums[mid]) {
            return -1
        }
        
        return mid
    }
}



class Solution {
    fun removeDuplicates(nums: IntArray): Int {
        var left = 0
        var right = 1
        var k = 1
        
        while (right < nums.size) {
            if (nums[left] == nums[right]) {
                right++
            } else {
                nums[++left] = nums[right++]
                k++
            }
        }
        
        return k
    }
}



// Broken binary search... again
/* The isBadVersion API is defined in the parent class VersionControl.
      fun isBadVersion(version: Int) : Boolean {} */

class Solution: VersionControl() {
    override fun firstBadVersion(n: Int) : Int {
        var left = 1
        var right = n
        var mid = (left + right) / 2
        
        while (left < right) {
            mid = (left + right) / 2
            if (isBadVersion(mid)) {
                right = mid - 1
            } else {
                left = mid + 1
            }
        }
        
        return mid
    }
}



// Another broken binary search
class Solution {
    fun searchInsert(nums: IntArray, target: Int): Int {
        var left = 0
        var right = nums.size - 1
        var mid = left + (right - left) / 2
        
        while (left < right) {
            if (nums[mid] == target) {
                return mid
            } else if (nums[right] == target) {
                return right
            } else if (mid == left) {
                if (target < nums[left]) {
                    return 0
                } else if (nums[left] < target && target < nums[right]) {
                    return right
                } else if (nums[right] < target) {
                    return right + 1
                }
            }
            
            mid = left + (right - left) / 2
        }
        
        // Shouldn't reach here
        return -1
    }
}



// Another broken binary search
class Solution {
    fun searchInsert(nums: IntArray, target: Int): Int {
        var left = 0
        var right = nums.size
        var mid = -1
        
        while (left < right) {
            mid = left + (right - left) / 2
            
            if (target < nums[mid]) {
                right = mid - 1
            } else if (target > nums[mid]) {
                left = mid + 1
            } else {
                return mid
            }
        }
        
        if (target < nums[mid]) {
            return 0
        } else if (target > nums[mid]) {
            return mid + 1
        } else {
            return mid
        }
    }
}



class Solution {
    fun searchInsert(nums: IntArray, target: Int): Int {
        var left = 0
        var right = nums.size
        var mid = left + (right - left) / 2
        
        while (left < right) {
            if (target < nums[mid]) {
                right = mid - 1
            } else if (target > nums[mid]) {
                left = mid + 1
            } else {
                return mid
            }
            
            mid = left + (right - left) / 2
        }
        
        println("target: " + target)
        println("left: " + left)
        println("right: " + right)
        println("mid: " + mid + "\n")
        
        if (target < nums[mid]) {
            return 0
        } else if (target > nums[mid]) {
            return mid + 1
        } else {
            return mid
        }
    }
}



// BROKEN WILL FIX ASAP
class Solution {
    fun reverseString(s: CharArray): Unit {
        var left = 0
        var right = s.size - 1
        while (left < right) {
            s[left] = s[right]
            left++;
            right--;
        }
    }
}



// SOLVED (with a little help)
class Solution {
    fun searchInsert(nums: IntArray, target: Int): Int {
        var l = 0
        var r = nums.size - 1
        
        while (l <= r) {
            var mid = l + ((r - l) / 2)
            
            if (nums[mid] == target) {
                return mid
            }
            
            if (nums[mid] < target) {
                l = mid + 1
            } else {
                r = mid - 1
            }
        }
        
        return l
    }
}



// SOLVED: BETTER ALGORITHM FOR FIRST BAD VERSION
class Solution: VersionControl() {
    override fun firstBadVersion(n: Int) : Int {
        var l = 1
        var r = n
        
        while (l <= r) {
            var mid = l + (r - l) / 2
            
            if (isBadVersion(mid)) {
                r = mid - 1
            } else {
                l = mid + 1
            }
        }
        
        return l
	}
}



class Solution {
    fun removeNthFromEnd(head: ListNode?, n: Int): ListNode? {
        var p1 = head
        var p2 = head!!.next
        var offset = 1
        
        while (p2!!.next != null) {
            if (offset < n) {
                offset++
            } else {
                if (p1!!.next!!.next == null) {
                    p1.next = null
                } else if (p2!!.next!!.next == null) {
                    p1.next = p1!!.next!!.next
                }
                p1 = p1!!.next
            }
            
            p2 = p2!!.next
        }
        
        return head
    }
}



class Solution {
    fun searchInsert(nums: IntArray, target: Int): Int {
        var l = 0
        var r = nums.lastIndex
        
        while (l <= r) {
            var mid = l + (r - l) / 2
            
            if (nums[mid] == target) {
                return mid
            } else if (nums[mid] < target) {
                l = mid + 1
            } else {
                r = mid - 1
            }
        }
        
        return l
    }
}



class Solution {
    fun sortedSquares(nums: IntArray): IntArray {
        if (nums.size == 1) {
            nums[0] *= nums[0]
            return nums
        }
        
        var newNums = IntArray(nums.size)
        var l = 0
        var r = 1
        var newNumPos = 0
        
        while(Math.abs(nums[r]) <= Math.abs(nums[l]) && r < newNums.lastIndex) {
            l++
            r++
        }
        
        while (0 <= l || r <= nums.lastIndex) {
            val lSquared = if (0 <= l) nums[l] * nums[l] else -1
            val rSquared = if (r <= nums.lastIndex) nums[r] * nums[r] else -1
            
            if ((lSquared <= rSquared && lSquared > -1) || rSquared < 0) {
                newNums[newNumPos++] = lSquared
                l--
            } else if ((rSquared < lSquared && rSquared > -1) || lSquared < 0) {
                newNums[newNumPos++] = rSquared
                r++
            }
        }
        
        return newNums;
    }
}



class Solution {
    fun rotate(nums: IntArray, k: Int): Unit {
        if (nums.size == 1 || k == 0 || k % nums.size == 0) {
            return
        }
        
        var p1 = nums.size - (k % nums.size)
        var p2 = 0
        var newNums = IntArray(nums.size)
        
        while (p2 < newNums.size) {
            newNums[p2++] = nums[p1++];
            
            if (p1 > nums.lastIndex) {
                p1 = 0
            }
        }
        
        for (i in nums.indices) {
            nums[i] = newNums[i]
        }
    }
}



class Solution {
    fun moveZeroes(nums: IntArray): Unit {
        nums.sort()
        
        var l = 0
        var r = firstNonZeroPos(nums)
        
        val noNonZeroPositionsFound = if (r < 0) true else false
        if (noNonZeroPositionsFound) {
            return
        }
        
        while (r < nums.size) {
            if (nums[l] == 0) {
                swap(l, r, nums)
            }
            l++
            r++
        }
    }
    
    fun swap(i: Int, j: Int, nums: IntArray) {
        var temp = nums[i]
        nums[i] = nums[j]
        nums[j] = temp
    }
    
    fun firstNonZeroPos(nums: IntArray): Int {
        for (i in nums.indices) {
            if (nums[i] != 0) {
                return i
            }
        }
        
        return -1
    }
}



// TIME LIMITED EXCEEDED IMPLEMENTATION
import java.util.LinkedList

class Solution {
    fun updateMatrix(mat: Array<IntArray>): Array<IntArray> {
        var i = 0
        var j = 0
        val newMat = mat.copyOf()
        
        for (i in mat.indices) {
            for (j in mat[i].indices) {
                var curVertex = mat[i][j]
                if (curVertex == 1) {
                    newMat[i][j] = getDistanceToClosestZero(i, j, mat)
                } else {
                    newMat[i][j] = 0
                }
            }
        }
        return newMat
    }
    
    private fun getDistanceToClosestZero(i: Int, j: Int, mat: Array<IntArray>): Int {
        val queue = LinkedList<GraphNode>() // Elements will have i, j, and distance from start
        val visitedMat = Array<IntArray>(mat.size, { i -> IntArray(mat[i].size, { j })})
        for (i in visitedMat.indices) {
            for (j in visitedMat[i].indices) {
                visitedMat[i][j] = 0
            }
        }
        
        queue.add(GraphNode(mat[i][j], i, j, 0))
        
        while (!queue.isEmpty()) {
            val nextVertex = queue.poll()
            val nextVertexIsVisited = visitedMat[nextVertex.i][nextVertex.j] == 1
            if (!nextVertexIsVisited) {
                 if (nextVertex.value == 0) {
                    return nextVertex.distanceFromStart
                } else {
                    visitedMat[nextVertex.i][nextVertex.j] = 1
                    markNeighbors(nextVertex.i, nextVertex.j, mat, queue, nextVertex.distanceFromStart)
                }
            }
        }
        
        // Shouldn't reach this line as there will always be at least one zero
        return -1
    }
    
    private fun markNeighbors(i: Int, j: Int, mat: Array<IntArray>, queue: LinkedList<GraphNode>, curDistanceFromStart: Int) {
        val up = i - 1
        val down = i + 1
        val left = j - 1
        val right = j + 1
        
        if (left > -1) {
            queue.add(GraphNode(mat[i][left], i, left, curDistanceFromStart + 1))
        }
        if (right < mat[i].size) {
            queue.add(GraphNode(mat[i][right], i, right, curDistanceFromStart + 1))
        }
        if (up > -1) {
            queue.add(GraphNode(mat[up][j], up, j, curDistanceFromStart + 1))
        }
        if (down < mat.size) {
            queue.add(GraphNode(mat[down][j], down, j, curDistanceFromStart + 1))
        }
    }
}

data class GraphNode(val value: Int, val i: Int, val j: Int, val distanceFromStart: Int)



class Solution {
    fun reverseString(s: CharArray): Unit {
        var left = 0
        var right = s.size - 1
        while (left < right) {
            val temp = s[left]
            s[left] = s[right]
            s[right] = temp
            left++
            right--
        }
    }
}



class Solution {
    fun reverseWords(s: String): String {
        val sb = StringBuilder()
        val words = s.split(" ")
        var index = 0
        words.forEach {
            if (index != words.lastIndex) {
                sb.append(it.reversed() + " ")
            } else {
                sb.append(it.reversed())
            }
            index++
        }
        return sb.toString()
    }
}



// Another freaking rotate array implementation
class Solution {
    fun rotate(nums: IntArray, k: Int): Unit {
        var p1 = nums.size - k
        var p2 = 0
        var newNums = IntArray(nums.size)
        
        while (p2 < nums.size) {
            newNums[p2++] = nums[p1++]
            if (nums.size <= p1) {
                p1 = 0
            }
        }
        
        for (i in nums.indices) {
            nums[i] = newNums[i]
        }
    }
}



class Solution {
    fun moveZeroes(nums: IntArray): Unit {
        var p1 = 0
        var p2 = 0
        
        while (p2 < nums.size) {
            if (nums[p2] != 0) {
                swap(p1, p2, nums)
                p1++
            }
            p2++
        }
    }
    
    fun swap(i: Int, j: Int, nums: IntArray) {
        val temp = nums[i]
        nums[i] = nums[j]
        nums[j] = temp
    }
}



// Twosum O(nlogn) solution
class Solution {
    fun twoSum(numbers: IntArray, target: Int): IntArray {
        // O(nlogn) solution
        for (i in numbers.indices) {
            val difference = target - numbers[i]
            val indexOfDifference = binarySearch(numbers, i + 1, difference)
            
            if (indexOfDifference > -1) {
                return intArrayOf(i + 1, indexOfDifference + 1)
            }
        }
        
        // Shouldn't reach this because always exactly one solution
        return intArrayOf(-1, -1)
    }
    
    private fun binarySearch(numbers: IntArray, startIndex: Int, target: Int): Int {
        var l = startIndex
        var r = numbers.lastIndex
        
        while (l <= r) {
            val mid = l + (r - l) / 2
            
            if (numbers[mid] == target) {
                return mid
            }
            
            if (numbers[mid] < target) {
                l = mid + 1
            } else {
                r = mid - 1
            }
        }
        
        return -1
    }
}



// Very suboptimal O(k*n) solution
class Solution {
    fun rotate(nums: IntArray, k: Int): Unit {
        if (nums.size < 2 || k == 0) {
            return
        }
        
        for (rotation in 1..k) {
            shiftRight(nums)
        }
    }
    
    private fun shiftRight(nums: IntArray) {
        for (i in (nums.lastIndex) downTo 1) {
            swap(nums, i, i-1)
        }
    }
    
    private fun swap(nums: IntArray, i: Int, j: Int) {
        val temp = nums[i]
        nums[i] = nums[j]
        nums[j] = temp
    }
}



// Middle node
/**
 * Example:
 * var li = ListNode(5)
 * var v = li.`val`
 * Definition for singly-linked list.
 * class ListNode(var `val`: Int) {
 *     var next: ListNode? = null
 * }
 */
class Solution {
    fun middleNode(head: ListNode?): ListNode? {
        var i = 1
        var p1 = head
        var p2 = head
        
        while(p2 != null) {
            if (i % 2 == 0) {
                p1 = p1?.next
            }
            i++
            p2 = p2?.next
        }
        
        return p1
    }
}



// WIP: Codewars kata
package twotoone

fun longest(s1:String, s2:String):String {
    var charArray = charArrayOf()
    StringBuilder().append(s1).append(s2).toCharArray(charArray)
    return charArray.distinct().toString().reversed()
}



import kotlin.math.*;

class Solution {
    // you need treat n as an unsigned value
    fun reverseBits(n:Int):Int {
        return Integer.reverse(n)
    }
}



class Solution {
    fun containsDuplicate(nums: IntArray): Boolean {
        val seen = hashSetOf<Int>()
        for (num in nums) {
            if (seen.contains(num)) {
                return true
            } else {
                seen.add(num)
            }
        }
        return false
    }
}



// TODO: Debug, currently getting wrong answer
class Solution {
    fun maxArea(height: IntArray): Int {
        // Had to have some hints but here we go
        var p1 = 0
        var p2 = height.lastIndex
        var maxArea = 0
        
        while (p1 < p2) {
            if (height[p1+1] > height[p1]) {
                p1++
                maxArea = calcArea(p1, p2, height)
            }
            else if (height[p2-1] > height[p2]) {
                p2--
                maxArea = calcArea(p1, p2, height)
            }
            else {
                break
            }
        }
        
        return maxArea
    }
    
    private fun calcArea(p1: Int, p2: Int, height: IntArray): Int {
        return Math.abs(p2 - p1) * Math.max(height[p1], height[p2])
    }
}



// Faster solution to twoSum
class Solution {
    fun twoSum(nums: IntArray, target: Int): IntArray {
        var p1 = 0
        var p2 = nums.lastIndex

        while (p1 < p2) {
            val sum = nums[p1] + nums[p2]
            if (sum == target) {
                return intArrayOf(p1 + 1, p2 + 1)
            } else if (sum < target) {
                p1++
            } else {
                p2--
            }
        }
        
        // Shouldn't reach here because exactly one solution
        return intArrayOf(-1, -1)
    }
}



// maxArea optimal solution (had to look at answer for this one, still need to understand it better).
class Solution {
    fun maxArea(height: IntArray): Int {
        // Had to look at solution for this one
        var l = 0
        var r = height.lastIndex
        var maxArea = 0
        
        while (l < r) {
            maxArea = Math.max(maxArea, calcArea(l, r, height))
            if (height[l] < height[r]) {
                l++
            } else {
                r--
            }
        }
        
        return maxArea
    }
    
    private fun calcArea(l: Int, r: Int, height: IntArray): Int {
        return Math.abs(r - l) * Math.min(height[l], height[r])
    }
}



// Recursive In-order traversal
class Solution {
    fun inorderTraversal(root: TreeNode?): List<Int> {
        val ans = mutableListOf<Int>()
        traverse(root, ans)
        return ans
    }
    
    private fun traverse(root: TreeNode?, ans: MutableList<Int>) {
        if (root == null) return
        
        traverse(root!!.left, ans)
        ans.add(root!!.`val`)
        traverse(root!!.right, ans)
    }
}



// Terrible mess... will work on it later hopefully
import java.util.LinkedList

class Solution {
    fun isValid(s: String): Boolean {
        if (s.length % 2 != 0) {
            return false
        }
        // There are just two conditions we need to check
        // to make sure that each parentheses is closed in the right order
        // 1. Each open char must be closed.
        // 2. There cannot be an open char with a closed char of a different type immediately after
        var openParentheses = 0
        var openBraces = 0
        var openBrackets = 0
        for (i in 0..s.lastIndex) {
            val character = s[i]
            
            // 1. Make sure each char is matched
            when (character) {
                '(' -> openParentheses++
                '{' -> openBraces++
                '[' -> openBrackets++
                ')' -> openParentheses--
                '}' -> openBraces--
                ']' -> openBrackets--
            }
            
            // 2. Make sure each char is closed in correct order
            if (i < s.lastIndex - 1) {
                val nextCharacter = s[i + 1]
                // Because the string only consists of parentheses, I can do this
                val ans = isNextCharInOrder(character, nextCharacter)
                println(ans)
                if (!ans) return false
            }
        }
        
        return openParentheses == 0 && openBraces == 0 && openBrackets == 0
    }
    
    private fun isOpen(character: Char) = (character == '(' || character == '{' || character == '[')
    private fun inverseOf(character: Char): Char {
        return when (character) {
            '(' -> ')'
            '{' -> '}'
            '[' -> ']'
            ')' -> '('
            '}' -> '{'
            ']' -> '['
            else -> ' '
        }
    }
    private fun isNextCharInOrder(character: Char, nextCharacter: Char) = (isOpen(nextCharacter) || nextCharacter == inverseOf(character))
}



// Beginning stages of solution
class Solution {
    fun lengthOfLongestSubstring(s: String): Int {
        // Initial solution
        // 3 vars: curSubstringLength, maxSubstringLength, and curSubstring (hash set)
        var curSubstringLength = 0
        var maxSubstringLength = 0
        var curSubstring = hashSetOf<Char>()
        
        for (i in 0..s.length) {
            if (!curSubstring.contains(s[i])) {

            } else {
                if (maxSubstringLength < curSubstringLength) {
                    maxSubstringLength = curSubstringLength
                }
                curSubstring.clear()
            }

        }
    }
}



class Solution {
    fun validPalindrome(s: String): Boolean {
        if (s.length == 1) return true
        if (s.length == 2) return s[0] == s[1]
        var p1 = 0
        var p2 = s.length - 1
        
        while (p1 < p2) {
            var char1 = s[p1]
            var char2 = s[p2]
            if (char1 != char2) {
                if (p1 == p2 - 1) {
                    return true
                } else if (s[p1 + 1] == s[p2] || s[p2 - 1] == s[p1]) {
                    return true
                }
            }
            p1++
            p2--
        }
        
        return true
        // If s is palindrome ret true
        // If s is not palindrome but at most one char can be removed to make it a palindrome, ret true
        // If s is not palindrome and no single char can be removed to make it a palindrome, ret false
    }
}



// TOO TIRED
/**
 * Example:
 * var li = ListNode(5)
 * var v = li.`val`
 * Definition for singly-linked list.
 * class ListNode(var `val`: Int) {
 *     var next: ListNode? = null
 * }
 */
class Solution {
    fun deleteDuplicates(head: ListNode?): ListNode? {
        val seen = hashSetOf<Char>()
        var p1 = head
        while (p1 != null) {
            if (seen.contains(p1.`val`)) {
                var next = p1!!.next
                p1!!.next = p1?.next?.next
                p1 = next
            } else {
                seen.add(p1.`val`)
                p1 = p1?.next
            }
        }
        
        return head
    }
}



// Merge binary trees
/**
 * Example:
 * var ti = TreeNode(5)
 * var v = ti.`val`
 * Definition for a binary tree node.
 * class TreeNode(var `val`: Int) {
 *     var left: TreeNode? = null
 *     var right: TreeNode? = null
 * }
 */
class Solution {
    fun mergeTrees(root1: TreeNode?, root2: TreeNode?): TreeNode? {
        if (root1 == null) return root2
        if (root2 == null) return root1
        root1.`val` = root1.`val` + root2.`val`
        root1.left = mergeTrees(root1.left, root2.left)
        root1.right = mergeTrees(root1.right, root2.right)
        return root1
    }
}



// maxSubArray WIP
class Solution {
    fun maxSubArray(nums: IntArray): Int {
        var maxSum = nums[0]
        var curSum = 0
        for (i in nums.indices) {
            curSum += nums[i]
            if (maxSum < curSum) {
                maxSum = curSum
            } else if (curSum < 0) {
                curSum = 0
            }
        }
        return maxSum
    }
}



//maxSubArray - kadane's algo variation
class Solution {
    fun maxSubArray(nums: IntArray): Int {
        var maxSum = nums[0]
        var curSum = 0
        for (i in nums.indices) {
            curSum += nums[i]
            if (maxSum < curSum) {
                maxSum = curSum
            }
            if (curSum < 0) {
                curSum = 0
            }
        }
        return maxSum
    }
}



// addBinary WIP
class Solution {
    fun addBinary(a: String, b: String): String {
        var p1 = 0
        var p2 = 0
        var carry = 0
        
        while (p1 < a.length || p2 < b.length) {
            if (p1 < a.length && p2 < b.length) {
                
            }
        }
    }
    
    fun addBinaryChars(a: Char, b: Char): CharArray {
        when (a) {
            '0' -> return if (b == '0') charArrayOf('0', '0') else charArrayOf('1','0')
            '1' -> return if (b == '0') charArrayOf('1', '0') else charArrayOf('1','1')
        }
    }
}



// TOO TIRED
class Solution {
    fun addBinary(a: String, b: String): String {
        var p1 = a.lastIndex
        var p2 = b.lastIndex
        var carry = '0'
        var sb = StringBuilder()
        
        while (p1 > 0 || p2 > 0) {
            var digitA = if (p1 > 0) {
                p1--
                a[p1]
            } else {
                '0'
            }
            var digitB = if (p2 > 0) {
                p2--
                b[p2]
            } else {
                '0'
            }
            var resultA = addBinaryChars(digitA, digitB)
            var resultB = addBinaryChars(resultA[0], carry)
            sb.append(resultB[0])
            carry = resultB[1]
        }
        
        return sb.reverse().toString()
    }
    
    fun addBinaryChars(a: Char, b: Char): CharArray {
        // Returns tuple of output digit and carry value
        return when (a) {
            '0' -> if (b == '0') charArrayOf('0', '0') else charArrayOf('1','0')
            '1' -> if (b == '0') charArrayOf('1', '0') else charArrayOf('0','1')
            else -> charArrayOf('0', '0')
        }
    }
}



// addBinary WIP 2
class Solution {
    fun addBinary(a: String, b: String): String {
        var p1 = a.lastIndex
        var p2 = b.lastIndex
        var sb = StringBuilder()
        
        var carry = 0
        while (p1 > 0 || p2 > 0 || carry > 0) {
            var digitA = if (p1 > 0) a[p1 + 1].toInt() else 0
            var digitB = if (p2 > 0) b[p2 + 1].toInt() else 0
        
            var res = digitA + digitB + carry
            sb.append(res % 2)
            carry = // IDK
        }
        
        return sb.reverse().toString()
    }
}



// Two sum WIP
class Solution {
    fun twoSum(nums: IntArray, target: Int): IntArray {
        val sortedNums = nums.sorted()
        var p1 = 0
        var p2 = nums.lastIndex
        
        while (p1 < p2) {
            var sum = nums[p1] + nums[p2]
            
            if (sum == target) {
                return intArrayOf(p1, p2)
            }
            
            if (sum < target) {
                p1++
            } else {
                p2--
            }
        }
        
        return intArrayOf(p1, p2)
    }
}



// Working solution for longest prefix problem
class Solution {
    // Would probably be a good place for a trie data structure
    fun longestCommonPrefix(strs: Array<String>): String {
        if (strs.isEmpty() || strs[0].isEmpty()) return ""
        
        var stringPos = 0
        var i = 0
        var longestCommonPrefix = StringBuilder()
        var lastChar = strs[i][stringPos]

        // The second we hit the end of a word or the prefix differs, return
        while (stringPos < strs[i].length && strs[i][stringPos] == lastChar) {
            lastChar = strs[i][stringPos]
            i++
            if (strs.size <= i) {
                longestCommonPrefix.append(lastChar)
                stringPos++
                i = 0
                if (stringPos < strs[i].length)
                    lastChar = strs[i][stringPos]
            }
        }
        return longestCommonPrefix.toString()
    }
}



class Solution {
    fun permute(nums: IntArray): List<List<Int>> {
        // Brute force method:
        // Generate all possible permutations:
        // 1. Allocate an array of size n! that stores arrays that are n long
        // 2. Loop through each array and somehow figure out 
        //    how to replace the elements such that each permutation shows up.
        // 3. Currently I understand how many elements need to be allocated but I don't understand how.
        
        // Other method
        // Mathmatically determine how many 
        // [1, 2, 3, 4]
        // [1, 2, 4, 3]
        // [1, 3, 2, 4]
        // [1, 3, 4, 2]
        // [1, 4, 2, 3]
        // [1, 4, 3, 2]
        val newNums = Array<IntArray>(factorial(nums.size)) { permutation -> IntArray(nums.size, {0})}
        for (num in nums) {
            
        }
    }
    
    fun factorial(num: Int): Int {
        var result = 1
        for (i in 2..num) result *= i
        return result
    }

}