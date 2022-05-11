class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        result = []
        
        if len(nums) == 1:
            return [nums[:]]
        
        for i in range(len(nums)):
            n = nums.pop(i)
            perms = self.permute(nums)
            for perm in perms:
                print(perm)
                perm.append(n)
                
            nums.append(n)
            result.extend(nums)
            
        return result
        