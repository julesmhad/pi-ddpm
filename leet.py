from typing import List

def isPairSum (nums: list, target: int):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums [j] == target:
                print(f"Target is true at idx: {nums[i]+1}, {nums[j]+1} \nValues are: {nums[i]} , {nums[j]}")
                return True
    return False



def twoSum (nums:list, target:int):
    nums.sort()
    left, right = 0, len(nums) -1

    while left < right:
        c_sum = nums[left] + nums[right]
        if c_sum == target:
            print(f"Target is true at idx: {left+1}, {right+1} \nValues are: {nums[left]} , {nums[right]}")
            return True
            
        
        if c_sum < target:
            left += 1
        else:
            right -= 1
    
    return False


nums = [11, 3, 7, 5, 2]
a = twoSum(nums = nums, target =5)
print(a)
b = isPairSum(nums = nums, target =5)
print(b)


