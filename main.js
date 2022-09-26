function doubleChar(str) {
	var p1 = 0;
	var result = "";
	while (p1 < str.length) {
		result += str[p1];
		result += str[p1];
		p1++;
	}
	return result;
}

function betterThanAverage(classPoints, yourPoints) {
	return yourPoints > classPoints.reduce((a, b) => a + b, 0) / classPoints.length; 
  }

const rps = (p1, p2) => {
	if(p1 == p2)
		return "Draw!";
	else {
		if((p1=="rock"&&p2=="scissors")||(p1=="scissors"&&p2=="paper")||(p1=="paper"&&p2=="rock"))
			return "Player 1 won!";
		else 
			return "Player 2 won!";
	}
};



function plural(n) {
	return true
  }



function sumOfN(n) {
	var curSum = 0
	var result = []
	
	for (var i = 0; i <= Math.abs(n); i++) {
	  if (n > 0)
		curSum += i
	  else
		curSum -= i
	  result.push(curSum)
	}
	
	return result
  };



function toCamelCase(str){
	var strings = str.split('-').join('_').split('_');
	var result = "";
	for (var i = 0; i < strings.length; i++) {
		const curStr = strings[i];
		if (i > 0) {
		strings[i] = replaceAtIndex(curStr, 0, curStr[0].toUpperCase());
		}
	}
	return strings.reduce((prevVal, curVal) => prevVal + curVal, "");
}

function replaceAtIndex(str, index, replacement) {
	return str.substring(0, index) + replacement + str.substring(index + replacement.length);
}




// WIP 2
function montyHall(correctDoorNumber, participantGuesses) {
	var wins = 0;
	for (var i = 0; i < participantGuesses.length; i++) {
	  const participantGuess = participantGuesses[i];
	  const otherDoor = Math.min(Math.max(6 - (correctDoorNumber + participantGuess), 1), 3);
	  const doorToSwitchTo = Math.min(Math.max(6 - (otherDoor + participantGuess), 1), 3);
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
var search = function(nums, target) {
    let l = 0, r = nums.length - 1
    
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
};



// Pivot search
/**
 * @param {number[]} nums
 * @return {number}
 */
 var pivotIndex = function(nums) {
    var totalSum = 0;
    for (var i = 0; i < nums.length; i++) {
        totalSum += nums[i]
    }
    
    var runningSum = 0;
    for (var i = 0; i < nums.length; i++) {
        runningSum += nums[i];
        if ((runningSum - nums[i]) == (totalSum - runningSum)) {
            return i;
        }
    }
    return -1;
};



function buildString(...template){
	return `I like ${template.join(', ')}!`;
  }



  function isPangram(str){
	let alphabetArr = [];
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
  };



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
var binaryTreePaths = function(root) {
    if (root == null) {
        return
    }
    
    let pathsList = []
    let rootIsLeafNode = (root.left == null && root.right == null)
    if (rootIsLeafNode) {
        pathsList.push(root.val.toString())
    } else {
        pathsList.concat(binaryTreePaths(root.left), binaryTreePaths(root.right))
        for (let i = 0; i < pathsList.length; i++) {
            pathsList[i] = root.val.toString() + "->" + pathsList[i].toString()
        }
    }
    
    return pathsList
};

/**
 * @param {string[]} strs
 * @return {string}
 */
 let longestCommonPrefix = function(strs) {
    let output = "";
    let curCharIndex = 0;
    let curChar = strs[0][curCharIndex];
    
    while (true) {
        curChar = strs[0][curCharIndex];
        for (let currentWordIndex = 0; currentWordIndex < strs.length; currentWordIndex++) {
            let nextChar = strs[currentWordIndex][curCharIndex];
            if (nextChar != curChar) {
                return;
            }
        }
        output += curChar;
        curCharIndex++;
    }
    
    return output;
};

function squareDigits(num){
  //may the code be with you
  var numS = num.toString();
  var output = "";
  for (var i = 0; i < numS.length; i++) {
    output += parseInt(numS[i]) * parseInt(numS[i]);
  }
  return parseInt(output);
}

/**
 * @param {string} s
 * @return {boolean}
 */
 var isValid = function(s) {
  var openBrackets = []
  for (var i = 0; i < s.length; i++) {
      var curBracket = s[i];
      if (curBracket == '[' || curBracket == '{' || curBracket == '(') {
          openBrackets.push(curBracket);
      } else {
          // This where we return false because of unopened brackets.
          if (openBrackets.length < 1)
              return false;
          
          var lastOpenBracket = openBrackets[openBrackets.length - 1];
          if (inverseOf(lastOpenBracket) == curBracket) {
              openBrackets.pop();
          } else {
              // This is where we will return false because of incorrect bracket type or mismatched orders.
              return false;
          }
      }
  }
  
  // This is where we return false because of unclosed brackets.
  if (openBrackets.length == 0)
      return true;
  else
      return false;
};

var inverseOf = function(openBracket) {
  switch (openBracket) {
      case '[':
          return ']';
      case '{':
          return '}';
      case '(':
          return ')';
  }
}

function getCount(str) {
  let count = 0;
  
  for (let i = 0; i < str.length; i++) {
    if (isVowel(str[i])) {
      count++;
    }
  }
  
  return count;
}

function isVowel(character) {
  if (character == "a" || character == "e" || character == "i" || character == "o" || character == "u") {
    return true;
  } else {
    return false;
  }
}

function removeDuplicates(nums){
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
};



/**
 * @param {number[]} arr
 * @return {number[][]}
 */
 var minimumAbsDifference = function(nums) {
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
};

function sayHi(name) {
  console.log(
    "Hi "
    + name
    + "!"
  )
}