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