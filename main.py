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