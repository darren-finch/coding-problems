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