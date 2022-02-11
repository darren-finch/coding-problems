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



// HALF SOLVED - TODO
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
    fun addTwoNumbers(l1: ListNode?, l2: ListNode?): ListNode? {
        // In classic addition, from right to left: 
        // 1. Add everything in first column
        // 2. Add ones digit of that addition to final answer.
        // 3. Add tens place (if it exists) to carry val
        // 4. In next column, do the addition again, and this time, add the carry val
        // 5. Repeat until you finish adding sum of last column to final answer
        //
        // In a linked-list environment, you will have pointers to the elements you're trying to work with
        var pointer1 = l1
        var pointer2 = l2
        var ans = ListNode(0)
        
        while (pointer1 != null || pointer2 != null) {
            val sumResult = pointer1.`val` + pointer2.`val`
            val digitToAddToAns = sumResult % 10
            val amtToCarry = ((sumResult / 10) as String).split(".")[0]
            print(digitToAddedToAns)
            print(git)
        }
        
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