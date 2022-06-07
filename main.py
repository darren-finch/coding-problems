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
