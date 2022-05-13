class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        result = []
        
        if len(nums) == 1:
            return [nums[:]]
        
        for i in range(len(nums)):
            n = nums.pop(i)
            print(nums)
            perms = self.permute(nums)
            for perm in perms:
                perm.append(n)
            
            nums.append(n)
            result.extend(perms)
            
        return result
        