function doubleChar(str) {
	var p1 = 0
	var result = ""
	while (p1 < str.length) {
		result += str[p1]
		result += str[p1]
		p1++
	}
	return result
}

function betterThanAverage(classPoints, yourPoints) {
	return yourPoints > classPoints.reduce((a, b) => a + b, 0) / classPoints.length
}

const rps = (p1, p2) => {
	if (p1 == p2) return "Draw!"
	else {
		if (
			(p1 == "rock" && p2 == "scissors") ||
			(p1 == "scissors" && p2 == "paper") ||
			(p1 == "paper" && p2 == "rock")
		)
			return "Player 1 won!"
		else return "Player 2 won!"
	}
}

function plural(n) {
	return true
}

function sumOfN(n) {
	var curSum = 0
	var result = []

	for (var i = 0; i <= Math.abs(n); i++) {
		if (n > 0) curSum += i
		else curSum -= i
		result.push(curSum)
	}

	return result
}

function toCamelCase(str) {
	var strings = str.split("-").join("_").split("_")
	var result = ""
	for (var i = 0; i < strings.length; i++) {
		const curStr = strings[i]
		if (i > 0) {
			strings[i] = replaceAtIndex(curStr, 0, curStr[0].toUpperCase())
		}
	}
	return strings.reduce((prevVal, curVal) => prevVal + curVal, "")
}

function replaceAtIndex(str, index, replacement) {
	return str.substring(0, index) + replacement + str.substring(index + replacement.length)
}

// WIP 2
function montyHall(correctDoorNumber, participantGuesses) {
	var wins = 0
	for (var i = 0; i < participantGuesses.length; i++) {
		const participantGuess = participantGuesses[i]
		const otherDoor = Math.min(Math.max(6 - (correctDoorNumber + participantGuess), 1), 3)
		const doorToSwitchTo = Math.min(Math.max(6 - (otherDoor + participantGuess), 1), 3)
		if (doorToSwitchTo == correctDoorNumber) {
			wins++
		}
	}

	return Math.ceil((wins / participantGuesses.length) * 100)
}

// Binary search off the top
/**
 * @param {number[]} nums
 * @param {number} target
 * @return {number}
 */
var search = function (nums, target) {
	let l = 0,
		r = nums.length - 1

	while (l <= r) {
		const mid = Math.floor((l + r) / 2)

		if (nums[mid] == target) {
			return mid
		}

		if (target < nums[mid]) {
			r = mid - 1
		} else {
			l = mid + 1
		}
	}

	return -1
}

// Pivot search
/**
 * @param {number[]} nums
 * @return {number}
 */
var pivotIndex = function (nums) {
	var totalSum = 0
	for (var i = 0; i < nums.length; i++) {
		totalSum += nums[i]
	}

	var runningSum = 0
	for (var i = 0; i < nums.length; i++) {
		runningSum += nums[i]
		if (runningSum - nums[i] == totalSum - runningSum) {
			return i
		}
	}
	return -1
}

function buildString(...template) {
	return `I like ${template.join(", ")}!`
}

function isPangram(str) {
	let alphabetArr = []
	for (let i = 0; i < str.length; i++) {
		const charCode = str.charCodeAt(i)
		if (isLetter(charCode)) {
		}
	}
}

function isLetter(charCode) {
	return (charCode > 64 && charCode < 91) || (charCode > 96 && charCode < 123)
}

String.prototype.toJadenCase = function () {
	let words = this.split(" ")
	let res = ""
	for (let i = 0; i < words.length; i++) {
		let word = words[i]
		if (i > 0) {
			res += " "
		}
		res += word.substring(0, 1).toUpperCase() + word.substring(1, word.length)
	}
	return res
}

/**
 * Definition for a binary tree node.
 * function TreeNode(val, left, right) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.left = (left===undefined ? null : left)
 *     this.right = (right===undefined ? null : right)
 * }
 */
/**
 * @param {TreeNode} root
 * @return {string[]}
 */
var binaryTreePaths = function (root) {
	if (root == null) {
		return
	}

	let pathsList = []
	let rootIsLeafNode = root.left == null && root.right == null
	if (rootIsLeafNode) {
		pathsList.push(root.val.toString())
	} else {
		pathsList.concat(binaryTreePaths(root.left), binaryTreePaths(root.right))
		for (let i = 0; i < pathsList.length; i++) {
			pathsList[i] = root.val.toString() + "->" + pathsList[i].toString()
		}
	}

	return pathsList
}

/**
 * @param {string[]} strs
 * @return {string}
 */
let longestCommonPrefix = function (strs) {
	let output = ""
	let curCharIndex = 0
	let curChar = strs[0][curCharIndex]

	while (true) {
		curChar = strs[0][curCharIndex]
		for (let currentWordIndex = 0; currentWordIndex < strs.length; currentWordIndex++) {
			let nextChar = strs[currentWordIndex][curCharIndex]
			if (nextChar != curChar) {
				return
			}
		}
		output += curChar
		curCharIndex++
	}

	return output
}

function squareDigits(num) {
	//may the code be with you
	var numS = num.toString()
	var output = ""
	for (var i = 0; i < numS.length; i++) {
		output += parseInt(numS[i]) * parseInt(numS[i])
	}
	return parseInt(output)
}

/**
 * @param {string} s
 * @return {boolean}
 */
var isValid = function (s) {
	var openBrackets = []
	for (var i = 0; i < s.length; i++) {
		var curBracket = s[i]
		if (curBracket == "[" || curBracket == "{" || curBracket == "(") {
			openBrackets.push(curBracket)
		} else {
			// This where we return false because of unopened brackets.
			if (openBrackets.length < 1) return false

			var lastOpenBracket = openBrackets[openBrackets.length - 1]
			if (inverseOf(lastOpenBracket) == curBracket) {
				openBrackets.pop()
			} else {
				// This is where we will return false because of incorrect bracket type or mismatched orders.
				return false
			}
		}
	}

	// This is where we return false because of unclosed brackets.
	if (openBrackets.length == 0) return true
	else return false
}

var inverseOf = function (openBracket) {
	switch (openBracket) {
		case "[":
			return "]"
		case "{":
			return "}"
		case "(":
			return ")"
	}
}

function getCount(str) {
	let count = 0

	for (let i = 0; i < str.length; i++) {
		if (isVowel(str[i])) {
			count++
		}
	}

	return count
}

function isVowel(character) {
	if (character == "a" || character == "e" || character == "i" || character == "o" || character == "u") {
		return true
	} else {
		return false
	}
}

function removeDuplicates(nums) {
	k = 1

	let p1 = 0
	let p2 = 1

	while (p2 < nums.length) {
		if (nums[p1] != nums[p2]) {
			nums[++p1] = nums[p2]
			k++
		}

		p2++
	}

	return k
}

/**
 * @param {number[]} arr
 * @return {number[][]}
 */
var minimumAbsDifference = function (nums) {
	nums.sort()

	let p1 = 0
	let p2 = 1
	let minAbsDiff = nums[p2] - nums[p1]

	while (p2 < nums.length) {
		let curDiff = nums[p2] - nums[p1]
		if (curDiff < minAbsDiff) {
			minAbsDiff = curDiff
		}

		p1++
		p2++
	}

	let result = []
	p1 = 0
	p2 = 1

	console.log(minAbsDiff)

	while (p2 < nums.length) {
		if (nums[p2] - nums[p1] == minAbsDiff) {
			result.push([nums[p1], nums[p2]])
		}
		p1++
		p2++
	}

	return result
}

function fizzBuzz(n) {
	let result = new Array(n)
	for (let i = 1; i <= n; i++) {
		result[i - 1] = smallFizzBuzz(i)
	}
	return result
}

function smallFizzBuzz(n) {
	let s = ""
	if (n % 3 == 0) {
		s += "Fizz"
	}
	if (n % 5 == 0) {
		s += "Buzz"
	}
	if (n % 3 != 0 && n % 5 != 0) {
		s += n.toString()
	}

	return s
}

/**
 * @param {string} s
 * @return {number}
 */
function romanToInt(numeralArray) {
	let sum = 0
	let numeralIndex = 0

	while (numeralIndex < numeralArray.length) {
		let numeral = numeralArray[numeralIndex]
		let forwardAmount = 1
		switch (numeral) {
			case "I":
				if (numeralIndex < numeralArray.length - 1) {
					if (numeralArray[numeralIndex + 1] == "V") {
						sum += 4
						forwardAmount = 2
					} else if (numeralArray[numeralIndex + 1] == "X") {
						sum += 9
						forwardAmount = 2
					} else {
						sum += 1
					}
				} else {
					sum += 1
				}
				break
			case "X":
				if (numeralIndex < numeralArray.length - 1) {
					if (numeralArray[numeralIndex + 1] == "L") {
						sum += 40
						forwardAmount = 2
					} else if (numeralArray[numeralIndex + 1] == "C") {
						sum += 90
						forwardAmount = 2
					} else {
						sum += 10
					}
				} else {
					sum += 10
				}
				break
			case "C":
				if (numeralIndex < numeralArray.length - 1) {
					if (numeralArray[numeralIndex + 1] == "D") {
						sum += 400
						forwardAmount = 2
					} else if (numeralArray[numeralIndex + 1] == "M") {
						sum += 900
						forwardAmount = 2
					} else {
						sum += 100
					}
				} else {
					sum += 100
				}
				break
			case "V":
				sum += 5
				break
			case "L":
				sum += 50
				break
			case "D":
				sum += 500
				break
			case "M":
				sum += 1000
				break
		}
		numeralIndex += forwardAmount
	}

	return sum
}

var twoSum = function (nums, target) {
	for (let i = 0; i < nums.length; i++) {
		let neededDiff = target - nums[i]
		for (let j = 0; j < nums.length; j++) {
			if (i != j && nums[j] == neededDiff) {
				return [i, j]
			}
		}
	}
}

function howMuchILoveYou(nbPetals) {
	// your code
	let res = ""
	if (nbPatals % 2 == 0) {
		result += ""
	}
	return res
}

/**
 * Definition for singly-linked list.
 * function ListNode(val, next) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.next = (next===undefined ? null : next)
 * }
 */
/**
 * @param {ListNode} head
 * @param {number} n
 * @return {ListNode}
 */
function removeNthFromEnd(head, n) {
	// [1, 2, 3, 4, 5]
	// n = 5

	// [1, 2, 3, 4]
	// n = 2

	// [1]
	// n = 1

	// [1, 2]
	// n = 2

	// Brute force
	// Using two pointer approach, one pointer goes all the way to the end, while the first pointer follows behind at a distance of n-1. Then somehow we do more magic. I'm too tired.

	// Hypothetical algorithm
	// 1. Using two pointers, start both pointers at the first element, then run the second pointer thru the array until it's looked at n positions.
	let p1 = head
	let p2 = head
	let p1Distance = 1
	let listSize = 1

	while (p2 != null) {
		p2 = p2.next
		listSize++
	}

	while (p1Distance < n - 1) {
		p1 = p1.next
	}

	console.log("P1: " + p1)
	console.log("P2: " + p2)

	return head
}

function digPow(n, p) {
	let nAsStr = n.toString()
	let curPow = p
	let sum = 0

	for (let i = 0; i < nAsStr.length; i++) {
		let curDigit = nAsStr[i]
		sum += Math.pow(parseInt(curDigit), curPow++)
	}

	if (sum % n == 0) {
		return sum / n
	} else {
		return -1
	}
}

function howMuchILoveYou(nbPetals) {
	let phrases = ["I love you", "a little", "a lot", "passionately", "madly", "not at all"]

	return phrases[(nbPetals - 1) % phrases.length]
}

function filterString(value) {
	return parseInt(
		value
			.match(/[\d]*/g)
			.filter((character) => character != "")
			.join(""),
		10
	)
}

function sundayHelloWorld() {
	console.log("Hello Sunday, and hello world!")
}

/**
 * @param {number[]} nums
 * @param {number} target
 * @return {number}
 */
function searchInsert(nums, target) {
	let left = 0
	let right = nums.length - 1

	while (left <= right) {
		mid = (left + right) / 2

		if (target < nums[mid]) {
			right = mid - 1
		} else if (nums[mid] < target) {
			left = mid + 1
		} else if (nums[mid] == target) {
			return mid
		}
	}

	return left
}

/**
 * @param {string} s
 * @param {string} t
 * @return {boolean}
 */
function isIsomorphic(s, t) {
	let charMap = {}

	// Algo:
	// 1. Check through each character of both strings, check to
}

/**
 * Definition for a binary tree node.
 * function TreeNode(val, left, right) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.left = (left===undefined ? null : left)
 *     this.right = (right===undefined ? null : right)
 * }
 */
/**
 * @param {TreeNode} p
 * @param {TreeNode} q
 * @return {boolean}
 */
function isSameTree(p, q) {
	if (p == null && q == null) {
		return true
	}
	if ((p == null && q != null) || (q == null && p != null)) {
		return false
	}
	return p.val == q.val && isSameTree(p.left, q.left) && isSameTree(p.right, q.right)
}

/**
 * @param {number} numRows
 * @return {number[][]}
 */
function generate(numRows) {
	let result = []
	for (let n = 0; n < numRows; n++) {
		let row = []
		for (let k = 0; k <= n; k++) {
			row.push(factorial(n) / (factorial(n - k) * factorial(k)))
		}
		result.push(row)
	}
	return result
}

function factorial(n) {
	let mem = []
	return factorialWithMemoization(n, mem)
}

function factorialWithMemoization(n, mem) {
	if (n <= 1) {
		return 1
	} else if (mem[n] != null) {
		return mem[n]
	} else {
		mem[n] = n * factorial(n - 1, mem)
		return mem[n]
	}
}

function choose(n, k) {
	if (k > n) {
		return 0
	}

	return factorial(n) / (factorial(k) * factorial(n - k))
}

function factorial(n) {
	let result = 1
	for (let i = 1; i <= n; i++) {
		result *= i
	}
	return result
}

/**
 * @param {number[]} nums
 * @return {boolean[]}
 */
var prefixesDivBy5 = function (nums) {
	let result = []

	for (let i = 0; i < nums.length; i++) {
		let curNum = 0

		for (let j = i; j >= 0; j--) {
			if (nums[j] == 1) {
				curNum += Math.pow(2, i - j)
			}
		}

		if (curNum % 5 == 0) {
			result[i] = true
		} else {
			result[i] = false
		}
	}

	return result
}

// V2 USING BIT SHIFT
/**
 * @param {number[]} nums
 * @return {boolean[]}
 */
var prefixesDivBy5 = function (nums) {
	let result = []
	let actualNum = 0

	for (let i = nums.length - 1; i >= 0; i--) {
		if (nums[i] == 1) {
			actualNum += Math.pow(2, nums.length - i - 1)
		}
	}

	for (let j = nums.length - 1; j >= 0; j--) {
		result[j] = actualNum % 5 == 0
		actualNum = actualNum >>> 1
	}

	return result
}

/**
 * @param {number[]} nums
 * @return {boolean[]}
 */
var prefixesDivBy5 = function (nums) {
	let result = []
	let actualNum = 0

	for (let i = nums.length - 1; i >= 0; i--) {
		if (nums[i] == 1) {
			actualNum += Math.pow(2, nums.length - i - 1)
		}
	}

	for (let j = nums.length - 1; j >= 0; j--) {
		console.log("Actual num base-10: " + actualNum)
		console.log("Actual num base-2: " + actualNum.toString(2) + "\n")
		result[j] = actualNum % 5 == 0
		actualNum = Math.trunc(actualNum / 2)
	}

	return result
}

/**
 * @param {number[]} nums
 * @param {number} target
 * @return {number[]}
 */
var searchRange = function (nums, target) {
	let result = [-1, -1]
	let left = 0
	let right = nums.length - 1

	while (left <= right) {
		let mid = Math.floor((left + right) / 2)

		if (target == nums[mid]) {
			if (mid == 0 || target != nums[mid - 1]) {
				result[0] = mid
				break
			} else {
				right = mid - 1
			}
		} else if (target < nums[mid]) {
			right = mid - 1
		} else {
			left = mid + 1
		}
	}

	left = 0
	right = nums.length - 1

	while (left <= right) {
		let mid = Math.floor((left + right) / 2)

		if (target == nums[mid]) {
			if (mid == nums.length - 1 || target != nums[mid + 1]) {
				result[1] = mid
				break
			} else {
				left = mid + 1
			}
		} else if (target < nums[mid]) {
			right = mid - 1
		} else {
			left = mid + 1
		}
	}

	return result

	// 1. on pass 1, perform regular binary search
	// 2. once target is found, determine if it is the first element in its little group
	// 3. if it is, plug it into result and break loop
	// 4. if it is not, keep binary searching until step #2 is true or until you hit left (which will be the leftmost element in the target group in the worst case.)
	// 5. on pass 2, do the same steps as 1-4 but reversed
}

/**
 * @param {number[]} nums
 * @param {number} target
 * @return {number}
 */
var search = function (nums, target) {
	// Strategy:    if left < right then continue regular binary search
	//              else calculate mid and determine which side the target must lie on
	let left = 0
	let right = nums.length - 1
}

/**
 * @param {number[]} nums
 * @param {number} target
 * @return {number}
 */
var search = function (nums, target) {
	// Strategy:    use modified binary search to find k, then binary search the two halves
	// Time complexity: O(3logn) = O(logn)

	let left = 0
	let right = nums.length - 1

	while (left <= right) {
		let mid = (left + right) / 2

		if (nums[right] < nums[left]) {
		}
	}
}

// ********************************* WIP: MULTIPLICATIVE INVERSE ALGORITHM ******************************************
// Problem: Find the multiplicative inverse of num under mod n if num and n are coprime, else return -1.

function multiplicativeInverse(num, n) {
	// 1. Compute GCD of num and n using Euclid's Algorithm, saving the results along the way
	// 2. Using the Extended Euclidean Algorithm, find the two coefficients s and t such that 1 = s(num) + tn
	// 3. Return s mod n
	let gcdWithEuclideanResults = gcdWithEuclideanResults(num, n) // in production it'd probably be good to separate this out.
	let gcd = gcdWithEuclideanResults[0]
	if (gcd !== 1) {
		return -1
	}
	let euclideanResults = gcdWithEuclideanResults[1]

	// CONTINUEs
}

// Returns an array arr with arr[0] = GCD(x, y) and arr[1] = results of Euclidean algorithm
function gcdWithEuclideanResults(x, y) {
	if (y < x) {
		let temp = x
		x = y
		y = temp
	}

	let r = y % x

	let euclideanResults = [y, x]

	while (r !== 0) {
		euclideanResults.push(r)
		y = x
		x = r
		r = y % x
	}

	return [x, euclideanResults]
}

// Given x and n that are coprime, this algorithm finds the two coefficients s and t such that 1 = sx + tn
function extendedEuclideanAlgorithm(x, n, euclideanResults) {}

/**
 * @param {number} dividend
 * @param {number} divisor
 * @return {number}
 */
// THIS PROBLEM PROBABLY NEEDS FAST-EXPONENTIATION
function divide(dividend, divisor) {
	// 1. Divison Algorithm: dividend = quotient (q) * + remainder (r), where r < divisor
	// 2. If sign of dividend != sign of divisor, take the absolute value of both the dividend and divisor. Then let r = dividend - divisor. Repeatedly subtract divisor from dividend until dividend < divisor. Return the number of times you had to perform the subtraction, k, multiplied by -1
	// 3. If sign of dividend == sign of divisor, take the absolute value of both the dividend and divisor. Then let r = dividend - divisor. Repeatedly subtract divisor from dividend until dividend < divisor. Return the number of times you had to perform the subtraction, k.
	const dividendAndDivisorHaveSameSign = (dividend >= 0 && divisor > 0) || (dividend <= 0 && divisor < 0)

	dividend = Math.abs(dividend)
	divisor = Math.abs(divisor)

	let quotient = 0
	let remainder = dividend

	while (remainder >= divisor) {
		remainder -= divisor
		quotient++
	}

	if (!dividendAndDivisorHaveSameSign) {
		quotient *= -1
	}

	const maxInt32Value = Math.pow(2, 31) - 1
	const minInt32Value = Math.pow(-2, 31)
	if (quotient > maxInt32Value) {
		quotient = maxInt32Value
	} else if (quotient < minInt32Value) {
		quotient = minInt32Value
	}

	return quotient
}

function helloWorld() {
	console.log("Hello World!")
}

function lengthOfLongestSubstring() {
	let maxLength = 0
	const seenChars = new Set()

	for (let i = 0; i < s.length; i++) {
		let curLength = 0
		for (let j = i; j < s.length; j++) {
			const curChar = s[j]
			if (seenChars.has(curChar)) {
				break
			} else {
				curLength++
				seenChars.add(curChar)
			}
		}

		if (maxLength < curLength) {
			maxLength = curLength
		}
		seenChars.clear()
		curLength = 0
	}

	return maxLength
}

function recursiveFastPow(base, exponent) {
	if (exponent <= 0) return 1
	let result = Math.pow(recursiveFastPow(base, Math.floor(exponent / 2)), 2)
	if (exponent % 2 === 1) result *= base
	return result
}

function iterativeFastPow(base, exponent) {
	let result = 1

	while (exponent > 0) {
		let exponentIsOdd = exponent % 2 === 1
		if (exponentIsOdd) result *= base
		base *= base
		exponent = Math.floor(exponent / 2)
	}

	return result
}

function generateParenthesis(n) {
	// To build a recursive relation:
	// 1. Start with the base case
	// 2. If you have everything below you computed properly, what do you need to compute the case above you?
	// For this problem:
	// 1. Use backtracking to build up branches of a decision tree.
}

function helloSleep() {
	console.log("Brah I'm so tireds")
}

/**
 * @param {number} x
 * @param {number} n
 * @return {number}
 */
function myPow(originalBase, originalExponent) {
	let result = 1
	let base = originalBase
	let exponent = Math.abs(originalExponent)

	while (0 < exponent) {
		if (exponent % 2 == 1) result *= base
		base *= base
		exponent = Math.floor((exponent /= 2))
	}

	if (0 <= originalExponent) {
		return result
	} else {
		return 1 / result
	}
}

// This works but doesn't pass submission
/**
 * @param {number} n
 * @return {string[]}
 */
function generateParenthesis(n) {
	if (n === 1) return ["()"]

	let prevGeneratedParenthesis = generateParenthesis(n - 1)
	let results = []

	for (let u of prevGeneratedParenthesis) {
		let newlyGeneratedParenthesis = ["(" + u + ")", u + "()"]

		if ("()" + u !== u + "()") {
			newlyGeneratedParenthesis.push("()" + u)
		}

		results = results.concat(newlyGeneratedParenthesis)
	}

	return results
}

/**
 * @param {number[]} nums
 * @return {number[][]}
 */
function permute(nums) {
	if (nums.length === 1) return [nums]

	let permutations = []
	for (let curNum of nums) {
		let numsWithoutCurNum = nums.filter((num) => num != curNum)
		let permutationsOfNumsWithoutCurNum = permute(numsWithoutCurNum)
		let permutationsOfNumsForCurNum = permutationsOfNumsWithoutCurNum.map((permutation) => [curNum, ...permutation])
		permutations.push(...permutationsOfNumsForCurNum)
	}

	return permutations
}

/**
 * @param {number} m
 * @param {number} n
 * @return {number}
 */
function uniquePaths(m, n) {
	// Has something to do with counting the number of paths left at each tile.
}

function totalIncDec(x) {
	// The question seems that it can be phrased as the following:
	//
	// How many ways can you arrange the digits in a number
	// with x digits such that the entire number is either increasing or decreasing?
}

// Dirty but works attempt at generating combinations iteratively
let justHitMax = false

function combine(n, k) {
	let curCombination = []
	for (let i = 0; i < k; i++) {
		curCombination.push(i + 1)
	}

	let combinationCount = factorial(n) / (factorial(k) * factorial(n - k))
	let results = []
	for (let i = 1; i <= combinationCount; i++) {
		results.push([...curCombination])
		curCombination = generateNextCombination(curCombination, n, k)
	}

	return results
}

function generateNextCombination(curCombination, combinationRange, combinationLen) {
	let largestIndexNotAtMax = combinationLen - 1
	while (largestIndexNotAtMax >= 0) {
		let maxNumAtIndex = combinationRange - (combinationLen - largestIndexNotAtMax) + 1
		if (curCombination[largestIndexNotAtMax] < maxNumAtIndex) {
			break
		}
		largestIndexNotAtMax--
	}

	curCombination[largestIndexNotAtMax]++

	if (justHitMax) {
		let next = curCombination[largestIndexNotAtMax]
		for (let i = largestIndexNotAtMax + 1; i <= combinationLen - 1; i++) {
			curCombination[i] = next + (i - largestIndexNotAtMax)
		}
	}

	let maxForLargestIndexNotAtMax = combinationRange - (combinationLen - largestIndexNotAtMax) + 1

	justHitMax = curCombination[largestIndexNotAtMax] >= maxForLargestIndexNotAtMax

	return curCombination
}

function factorial(n) {
	if (n <= 1) return 1
	return n * factorial(n - 1)
}

// WIP: nextSubsetOfP where P_r,n is the set of all non-empty subsets of {1,2,...,n} with r or fewer elements.
function nextSubsetOfP(curSubset, n, r) {
	let largestIndexLessThanN = -1
	for (let i = 0; i < curSubset.length; i++) {
		if (curSubset[i] < n) {
			largestIndexLessThanN = i
		}
	}

	if (largestIndexLessThanN === -1) {
		return curSubset
	}

	if (curSubset.length < r && largestIndexLessThanN === r - 1) {
		return [...curSubset, curSubset[largestIndexLessThanN]++]
	} else {
		curSubset[largestIndexLessThanN]++
		let returnVal = []
		for (let i = 0; i <= largestIndexLessThanN; i++) {
			returnVal.push(curSubset[i])
		}
		return returnVal
	}
}

console.log(nextSubsetOfP([1, 2, 3], 4, 4))

/**
 * Definition for singly-linked list.
 * function ListNode(val, next) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.next = (next===undefined ? null : next)
 * }
 */
/**
 * @param {ListNode} list1Head
 * @param {ListNode} list2Head
 * @return {ListNode}
 */
var mergeTwoLists = function (list1Head, list2Head) {
	if (list1Head === null) return list2Head
	if (list2Head === null) return list1Head

	let head = null
	let curPointer = null
	let oppNext = null
	if (list1Head.val <= list2Head.val) {
		head = list1Head
		curPointer = list1Head
		oppNext = list2Head
	} else {
		head = list2Head
		curPointer = list2Head
		oppNext = list1Head
	}

	while (oppNext != null) {
		if (curPointer.next !== null && curPointer.next.val <= oppNext.val) {
			curPointer = curPointer.next
		} else {
			let temp = curPointer.next
			curPointer.next = oppNext
			oppNext = temp
		}
	}

	return head
}

// 3SUM BRUTE FORCE WIP
/**
 * @param {number[]} nums
 * @return {number[][]}
 */
function threeSum(nums) {
	// Stopped at needing to find a pattern in the solutions that could provide a clue as to how to solve this problem in a more efficient time complexity than O(n^3 + m) = O(n^3), and hopefully a better space complexity than O(2k) = O(k) where n is the size of the input, m is the number of triplets (not necessarily distinct) that add to 0, and k is the number of distinct triplets that add to 0.

	return threeSumBruteForce(nums)
}

function threeSumBruteForce(nums) {
	// Strategy: Find all triplets the obvious O(n^3) way, then filter out all duplicates
	let results = []
	let distinctResults = new Set()
	for (let i = 0; i < nums.length; i++) {
		for (let j = 0; j < nums.length; j++) {
			for (let k = 0; k < nums.length; k++) {
				if (i !== j && i !== k && j !== k && nums[i] + nums[j] + nums[k] === 0) {
					let sortedResult = [nums[i], nums[j], nums[k]].sort()
					if (!distinctResults.has(sortedResult)) {
						distinctResults.add(sortedResult)
					}
				}
			}
		}
	}

	// How the heck do you return a set as a list?
	// This would help tremendously!
	return [...distinctResults.entries()]
}

/**
 * @param {character[][]} board
 * @return {boolean}
 */
var isValidSudoku = function (board) {
	let columnHashSets = [
		new Set(),
		new Set(),
		new Set(),
		new Set(),
		new Set(),
		new Set(),
		new Set(),
		new Set(),
		new Set(),
	]
	let rowHashSets = [
		new Set(),
		new Set(),
		new Set(),
		new Set(),
		new Set(),
		new Set(),
		new Set(),
		new Set(),
		new Set(),
	]
	let subGridHashSets = [
		new Set(),
		new Set(),
		new Set(),
		new Set(),
		new Set(),
		new Set(),
		new Set(),
		new Set(),
		new Set(),
	]

	for (let row = 0; row < board.length; row++) {
		for (let column = 0; column < board.length; column++) {
			let cellEntry = board[row][column]
			let gridNum = 3 * Math.floor(row / 3) + Math.floor(column / 3)
			if (cellEntry !== ".") {
				if (
					!columnHashSets[column].has(cellEntry) &&
					!rowHashSets[row].has(cellEntry) &&
					!subGridHashSets[gridNum].has(cellEntry)
				) {
					columnHashSets[column].add(cellEntry)
					rowHashSets[row].add(cellEntry)
					subGridHashSets[gridNum].add(cellEntry)
				} else {
					return false
				}
			}
		}
	}

	return true
}

/**
 * @param {character[][]} board
 * @return {boolean}
 * This solution was inspired by another LeetCode solution (particularly the clever use of the Set())
 */
var isValidSudoku = function (board) {
	let seen = new Set()
	for (let row = 0; row < board.length; row++) {
		for (let column = 0; column < board.length; column++) {
			let cellEntry = board[row][column]
			if (cellEntry !== ".") {
				let gridNum = 3 * Math.floor(row / 3) + Math.floor(column / 3)

				let columnEntry = `column ${column}: ${cellEntry}`
				let rowEntry = `row ${row}: ${cellEntry}`
				let subGridEntry = `subGrid ${gridNum}: ${cellEntry}`

				if (seen.has(columnEntry) || seen.has(rowEntry) || seen.has(subGridEntry)) {
					return false
				}

				seen.add(columnEntry)
				seen.add(rowEntry)
				seen.add(subGridEntry)
			}
		}
	}

	return true
}

/**
 * This is an implementation of a solution with sorted arrays.
 * @param {number[]} nums1
 * @param {number[]} nums2
 * @return {number[]}
 */
var intersect = function (nums1, nums2) {
	nums1.sort()
	nums2.sort()

	if (nums1.length < nums2.length) {
		let temp = nums1
		nums1 = nums2
		nums2 = temp
	}

	let p1 = 0
	let p2 = 0
	let results = []

	while (p1 < nums1.length && p2 < nums2.length) {
		if (nums1[p1] == nums2[p2]) {
			results.push(nums1[p1])
		} else if (nums1[p1] < nums2[p2]) {
			p1++
		} else {
			p2++
		}
	}

	return results
}

// Algorithm has (O(n^2) time complexity), won't pass the final LeetCode test because it's just a little to slow.
/**
 * @param {number[]} nums
 * @return {boolean}
 */
var find132pattern = function (nums) {
	if (nums.length < 3) return false

	for (let i = 0; i < nums.length; i++) {
		let theThreeOrTwoInTheSequence = null
		for (j = i + 1; j < nums.length; j++) {
			if (nums[i] < nums[j]) {
				if (theThreeOrTwoInTheSequence !== null && nums[j] < theThreeOrTwoInTheSequence) {
					return true
				} else {
					theThreeOrTwoInTheSequence = nums[j]
				}
			}
		}
	}

	return false
}

let numberToLetterMappings = {
	2: ["a", "b", "c"],
	3: ["d", "e", "f"],
	4: ["g", "h", "i"],
	5: ["j", "k", "l"],
	6: ["m", "n", "o"],
	7: ["p", "q", "r", "s"],
	8: ["t", "u", "v"],
	9: ["w", "x", "y", "z"],
}

/**
 * @param {string} digits
 * @return {string[]}
 */
function letterCombinations(digits) {
	// Base cases
	if (digits.length === 0) {
		return []
	}

	if (digits.length === 1) {
		return numberToLetterMappings[digits[0]]
	}

	// Recursive step
	let results = []
	let recursiveResults = letterCombinations(digits.substring(1))
	let lettersForThisRecursiveStep = numberToLetterMappings[digits[0]]
	for (let i = 0; i < lettersForThisRecursiveStep.length; i++) {
		let resultsForCurLetter = []
		for (let j = 0; j < recursiveResults.length; j++) {
			resultsForCurLetter.push(lettersForThisRecursiveStep[i] + recursiveResults[j])
		}
		results.push(...resultsForCurLetter)
	}

	return results
}

/**
 * @param {number[]} nums
 * @return {number}
 */
var longestConsecutive = function (nums) {
	// Temporary brute force algorithm:
	// Create a list to store some lists
	// Until we have iterated nums.length times:
	//      Find the minimum element in nums, call it x
	//      Find a list in the list of lists where the last element is less than x by exactly one and put x at the end of it
	//      If we find a list with a last element that is equal to x by exactly one, then skip this iteration of the loop and do nothing
	//      If we cannot find a list with either of the above conditions, put x onto its own list.
	// This process will result in a list of lists with consecutive elements inside. Return the length of the longest such list.
}

// WIP: hasCycle
/**
 * Definition for singly-linked list.
 * function ListNode(val) {
 *     this.val = val;
 *     this.next = null;
 * }
 */

/**
 * @param {ListNode} head
 * @return {boolean}
 */
var hasCycle = function (head) {
	if (pos == -1 || head == null || head.next == null) {
		return false
	}

	let curr = head.next
	let prev = head

	while (prev !== null) {
		if (curr !== null && curr.next !== null) {
			let temp = curr.next
			curr.next = prev
			prev = temp
		}
	}
}

/**
 * @param {string} s
 * @return {string}
 */
var longestPalindrome = function (s) {
	// This may be a spin-off of the maximum subarray problem that can be solved with Knuth's algorithm. I just wish I remember what it was.
	// let maxPalindromeLength = 0
}

/**
 * @param {number[]} nums
 * @param {number[]} index
 * @return {number[]}
 */
function createTargetArray(nums, index) {
	let target = []
	for (let i = 0; i < nums.length; i++) {
		target = insert(target, nums[i], index[i])
	}
	return target
}

/**
 * @param {number[]} numsArr
 * @param {number} num
 * @param {number} index
 * @return {number[]}
 */
function insert(numsArr, num, index) {
	if (numsArr.length < index) {
		numsArr.push(num)
		return numsArr
	}
	return numsArr.slice(0, index).concat(num, numsArr.slice(index, numsArr.length))
}

/**
 * Definition for a binary tree node.
 * function TreeNode(val, left, right) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.left = (left===undefined ? null : left)
 *     this.right = (right===undefined ? null : right)
 * }
 */
/**
 * @param {TreeNode} root
 * @param {number} targetSum
 * @return {boolean}
 */
function hasPathSum(root, targetSum) {
	if (root == null) {
		return false
	}

	if (targetSum - root.val == 0 && root.left == null && root.right == null) {
		return true
	}

	return hasPathSum(root.left, targetSum - root.val) || hasPathSum(root.right, targetSum - root.val)
}

/**
 * @param {number[]} candidates
 * @param {number} target
 * @return {number[][]}
 */
var combinationSum = function (candidates, target) {
	// Overview of algorithm:
	// Base case: Return null if target < 0, or return [] if target == 0
	//
	// Declare results array
	// Loop through array
	// For each element, add that element to each list returned by a recursive call to combinationSum(candidates, target - cur element), if the return value of that function is not null
	// Add the elements of the list obtained from the previous step onto the results array
	// Return results array

	if (target < 0) {
		return null
	}
	if (target == 0) {
		return []
	}

	// WIP
	let results = []
	for (let candidate of candidates) {
		let newTarget = target - candidate
		let combinationsThatSumToNewTarget = combinationSum(candidates, newTarget)
		if (combinationsThatSumToNewTarget != null) {
			if (combinationsThatSumToNewTarget.length == 0) {
				results.push([candidate])
			} else {
				let combinationsToAddToResults = combinationsThatSumToNewTarget.map((elem) => [candidate, ...elem])
				results.push(...combinationsToAddToResults)
			}
		}
	}
	return results
}

function isIsogram(str) {
	const seenNumbers = new Set()
	for (let i = 0; i < str.length; i++) {
		const curChar = str[i]
		if (seenNumbers.has(curChar.toLowerCase())) {
			return false
		} else {
			seenNumbers.add(curChar.toLowerCase())
		}
	}
	return true
}

/**
 * @param {number} m
 * @param {number} n
 * @return {number}
 */
function uniquePaths(m, n) {
	if (m == 1 || n == 1) return 1

	// Basic algorithm (based on a BFS. Basic idea is to show how many ways you can get to each node as you go along.)
	// Assign the value of 0 to each node except the top left node, which will be assigned the value of 1.
	// Declare a new queue
	// Put the top left node in the queue
	// For each node until we've reached the end node:
	//      Pull off the next node from the queue
	//      Add the node's value to each of the node's neighbors
	//      Continue until there are no more nodes left to visit
	// The bottom right node will now have the number of possible unique paths the robot can take.
	const grid = new Array(m).fill([]).map((row) => new Array(n).fill(0))
	grid[0][0] = 1

	const unvisitedPositions = []
	unvisitedPositions.push({ row: 0, col: 0 })

	let i = 1

	while (0 < unvisitedPositions.length) {
		const curPos = unvisitedPositions.shift()
		let curPosNextDown = { row: curPos.row + 1, col: curPos.col }
		let curPosNextRight = { row: curPos.row, col: curPos.col + 1 }

		if (curPosNextDown.row < m) {
			grid[curPosNextDown.row][curPosNextDown.col] += grid[curPos.row][curPos.col]

			// TODO: OPTOMIZE THIS
			if (!unvisitedPositions.some((pos) => pos.row == curPosNextDown.row && pos.col == curPosNextDown.col)) {
				unvisitedPositions.push(curPosNextDown)
			}
		}
		if (curPosNextRight.col < n) {
			grid[curPosNextRight.row][curPosNextRight.col] += grid[curPos.row][curPos.col]

			// TODO: OPTOMIZE THIS
			if (!unvisitedPositions.some((pos) => pos.row == curPosNextRight.row && pos.col == curPosNextRight.col)) {
				unvisitedPositions.push(curPosNextRight)
			}
		}
	}

	return grid[m - 1][n - 1]
}

/**
 * @param {number[]} nums
 * @return {void} Do not return anything, modify nums in-place instead.
 */
var nextPermutation = function (nums) {
	let i = -1
	for (let k = 0; k < nums.length - 1; k++) {
		if (nums[k] < nums[k + 1]) {
			i = k
		}
	}

	// There is no larger re-arrangement, reverse the array passed in
	if (i == -1) {
		reverse(nums, 0, nums.length - 1)
		return
	}

	let j = i
	for (let k = i + 1; k < nums.length; k++) {
		if (nums[i] < nums[k]) {
			j = k
		}
	}

	swap(nums, i, j)

	sort(nums, i + 1, nums.length - 1)
}

// Bubble sort, to speed up the algorithm maybe use selection sort?
function sort(nums, start, stop) {
	for (let i = start; i <= stop; i++) {
		for (let j = i + 1; j <= stop; j++) {
			if (nums[j] < nums[i]) {
				swap(nums, i, j)
			}
		}
	}
}

function reverse(nums, start, stop) {
	while (start < stop) {
		swap(nums, start, stop)
		start++
		stop--
	}
}

function swap(nums, i, j) {
	let temp = nums[i]
	nums[i] = nums[j]
	nums[j] = temp
}

/**
 * @param {number[]} nums1
 * @param {number[]} nums2
 * @return {number[]}
 */
var nextGreaterElement = function (nums1, nums2) {
	// Algorithm:
	// This algorithm sounds like a proper greedy algorithm.
	// WHat we need to do is keep finding the local next greaststl
}

const queue = []
let top = null

var MyStack = function () {}

/**
 * @param {number} x
 * @return {void}
 */
MyStack.prototype.push = function (x) {
	queue.push(x)

	let temp = null
	for (let i = 0; i < queue.length - 1; i++) {
		temp = queue.shift()
		queue.push(temp)
	}

	top = temp
}

/**
 * @return {number}
 */
MyStack.prototype.pop = function () {
	return queue.shift()
}

/**
 * @return {number}
 */
MyStack.prototype.top = function () {
	return top
}

/**
 * @return {boolean}
 */
MyStack.prototype.empty = function () {
	return queue.length == 0
}

/**
 * Your MyStack object will be instantiated and called as such:
 * var obj = new MyStack()
 * obj.push(x)
 * var param_2 = obj.pop()
 * var param_3 = obj.top()
 * var param_4 = obj.empty()
 */

/**
 * @param {number} n
 * @return {number[][]}
 */
var generateMatrix = function (n) {
	// WIP, will use mathmatical properties to solve in a reasonable amount of time. The brute force algorithm would generate all possible combinationsa and check each of them for validity. Or something like that.
}

/**
 * Definition for singly-linked list.
 * function ListNode(val, next) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.next = (next===undefined ? null : next)
 * }
 */
/**
 * Definition for a binary tree node.
 * function TreeNode(val, left, right) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.left = (left===undefined ? null : left)
 *     this.right = (right===undefined ? null : right)
 * }
 */
/**
 * @param {ListNode} head
 * @return {TreeNode}
 */
var sortedListToBST = function (head) {
	// WIP, need to learn how to create a balanced binary search tree
}
