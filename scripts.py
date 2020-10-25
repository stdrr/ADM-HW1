# Say "Hello, World!" With Python

print('Hello, World!')

# Python If-Else

import math
import os
import random
import re
import sys

if __name__ == '__main__':
    n = int(input().strip())

    if n%2 == 1: # n is odd
        print('Weird')
    else:
        if 2 <= n <= 5:
            print('Not Weird')
        elif n <= 20: # we can omitt to check the lower bound thanks to the assumption that n is an integer
            print('Weird')
        else: 
            print('Not Weird')

# Arithmetic Operators

if __name__ == '__main__':
    a = int(input())
    b = int(input())

    print( a + b )
    print( a - b )
    print( a * b )

# Python: Division

if __name__ == '__main__':
    a = int(input())
    b = int(input())

    if b != 0:
        print( a // b )
        print( a / b )

# Loops

if __name__ == '__main__':
    n = int(input())

    for i in range(0,n): # loops until n-1
        print(i**2)

# Write a function

def is_leap(year):
    return (year%4 == 0 and year%100 != 0) or (year%400 == 0) 

year = int(input())
print(is_leap(year))

# Print Function

if __name__ == '__main__':
    n = int(input())

    for i in range(1,n+1):
        print(i, end='')

# List Comprehensions 

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())

    l = [ [i,j,k] for i in range(0,x+1) for j in range(0,y+1) for k in range(0,z+1) if i+j+k!=n ] # Remember: range(0,a) = [0,a[
    print(l)

# Find the Runner-Up Score!

if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())

    fst_max = snd_max = -100 # according to the constraints, the minimum value is -100
    for elem in arr:
        if fst_max < elem:
            snd_max = fst_max
            fst_max = elem
        elif fst_max != elem and snd_max < elem:
            snd_max = elem
    print(snd_max)

# Nested Lists

if __name__ == '__main__':
    students_lst = []
    for line in range(int(input())):
        name = input()
        score = float(input())
        students_lst.append([name, score])

    SCORE = 1
    fst_min = students_lst[0][SCORE]
    snd_min = 1000 # an arbitrary number bigger than the students' scores 
    for student in students_lst:
        if fst_min > student[SCORE]:
            snd_min = fst_min
            fst_min = student[SCORE]
        elif fst_min != student[SCORE] and snd_min > student[SCORE]:
            snd_min = student[SCORE]
    result = [student[0] for student in students_lst if student[SCORE] == snd_min]
    result.sort()
    for name in result:
        print(name)

# Finding the percentage

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    avg = sum(student_marks[query_name])/len(student_marks[query_name])
    print("%.2f"%avg)

# Lists

if __name__ == '__main__':
    N = int(input())
    l = []
    for i in range(N):
        op = input().split()
        if op[0] == 'append':
            l.append(int(op[1]))
        elif op[0] == 'insert':
            l.insert(int(op[1]), int(op[2]))
        elif op[0] == 'pop':
            l.pop()
        elif op[0] == 'print':
            print(l)
        elif op[0] == 'remove':
            l.remove(int(op[1]))
        elif op[0] == 'reverse':
            l.reverse()
        else: 
            l.sort()

# Tuples

if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())

    t = tuple(integer_list)
    print(hash(t))

# sWAP cASE

def swap_case(s):
    lst_str = list(s)
    for i in range(len(lst_str)):
        if lst_str[i].islower():
            lst_str[i] = lst_str[i].upper()
        elif lst_str[i].isupper():
            lst_str[i] = lst_str[i].lower()
    return ''.join(lst_str)

if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)

# String Split and Join

def split_and_join(line):
    return "-".join(line.split(" "))

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

# What's Your Name?

def print_full_name(first_name, last_name):
    print("Hello %s %s! You just delved into python."%(first_name, last_name))

if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)

# Mutations

def mutate_string(string, position, character):
    list_str = list(string)
    list_str[position] = character
    return ''.join(list_str)

if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)

# Find a string

def count_substring(string, sub_string):
    counter = 0
    index_found = -1
    cond = True
    while cond:
        index_found = string.find(sub_string, index_found+1)
        if index_found != -1:
            counter += 1
        else:
            cond = False
    return counter

if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()
    
    count = count_substring(string, sub_string)
    print(count)

# String Validators

if __name__ == '__main__':
    s = input()
    alpha = digit = lower = upper = False
    for substr in s:
        if substr.isalpha():
            alpha = True
        if substr.isdigit():
            digit = True
        if substr.islower():
            lower = True
        if substr.isupper():
            upper = True
    print( (alpha or digit) ) # the string has at least one alphanumeric character
    print(alpha) # the string has at least one alphabetic character
    print(digit) # the string has at least one digit
    print(lower) # the string has at least one lowercase character
    print(upper) # the string has at least one uppercase character

# Text Alignment

thickness = int(input()) #This must be an odd number
c = 'H'

for i in range(thickness): #Top Cone
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

for i in range(thickness+1): #Top Pillars
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

for i in range((thickness+1)//2): #Middle Belt
    print((c*thickness*5).center(thickness*6))    

for i in range(thickness+1): #Bottom Pillars
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

for i in range(thickness): #Bottom Cone
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

# Text Wrap

import textwrap

def wrap(string, max_width):
    return textwrap.fill(string, max_width)

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)

# Designer Door Mat

if __name__ == '__main__':
    dim = input().split()

    for i in range(1, int(dim[0]), 2): 
        print(('.|.'*i).center(int(dim[1]), '-'))
    print('WELCOME'.center(int(dim[1]), '-'))
    for i in range(int(dim[0])-2, 0, -2): 
        print(('.|.'*i).center(int(dim[1]), '-'))

# String Formatting

def print_formatted(number):
    padding = len(str(bin(number))[2:])
    for i in range(1, number+1):
        print('{} {} {} {}'.format(
            str(i).rjust(padding),
            str(oct(i))[2:].rjust(padding),
            str(hex(i))[2:].upper().rjust(padding),
            str(bin(i))[2:].rjust(padding)
        ))

if __name__ == '__main__':
    n = int(input())
    print_formatted(n)

# Alphabet Rangoli

import string

def print_rangoli(size):
    alphabet = string.ascii_lowercase
    width = 4 * size - 3
    s = ''
    for i in range(size, 0, -1):
        s += '-'*(2*(i-1))
        s += '-'.join(alphabet[i-1:size][::-1] + alphabet[i:size])
        s += '-'*(2*(i-1)) + '\n'
    print(s + s[:-width-2][::-1])

if __name__ == '__main__':
    n = int(input())
    print_rangoli(n)

# Capitalize!

import math
import os
import random
import re
import sys

def solve(s):
    s_lst = s.split(' ')
    for i in range(len(s_lst)):
        s_lst[i] = s_lst[i].capitalize()
    return ' '.join(s_lst)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    s = input()
    result = solve(s)
    fptr.write(result + '\n')
    fptr.close()

# The Minion Game

def minion_game(string):
    score = {'Kevin': 0, 'Stuart': 0}
    vowels = 'AEIOU'
    for i in range(len(string)):
        if string[i] in vowels:
            score['Kevin'] += len(string) - i   # since Kevin can make substrings
                                                # beginning with string[i] at most of 
                                                # len(string)-i
        else:
            score['Stuart'] += len(string) - i  
    if score['Kevin'] > score['Stuart']:
        print('Kevin {}'.format(score['Kevin']))
    elif score['Kevin'] < score['Stuart']:
        print('Stuart {}'.format(score['Stuart']))
    else:
        print('Draw')

if __name__ == '__main__':
    s = input()
    minion_game(s)

# Merge the Tools!

def merge_the_tools(string, k):
    for i in range(0, len(string), k):
        substring = ''
        for j in range(i, i+k):
            substring += string[j] if string[j] not in substring else ''
        print(substring)

if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)

# Introduction to Sets

def average(array):
    s = set(array)
    return sum(s)/len(s)

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)

# No Idea!

if __name__ == '__main__':
    n, m = input().split()
    l = input().split()
    A = set(input().split())
    B = set(input().split())

    hppness = 0
    for elem in l:
        if elem in A and elem not in B:
            hppness += 1
        elif elem in B and elem not in A:
            hppness -= 1
    print(hppness)

# Symmetric Difference

if __name__ == '__main__':
    M_len = input()
    M = set(map(int, input().split()))
    N_len = input()
    N = set(map(int, input().split()))

    sym_diff = M.difference(N).union(N.difference(M))
    sym_diff_lst = list(sym_diff)
    sym_diff_lst.sort()
    for elem in sym_diff_lst:
        print(elem)

# Set .add()

if __name__ == '__main__':
    N = int(input())
    d_countries = set()
    for i in range(N):
        d_countries.add(input())
    print(len(d_countries))

# Set .discard(), .remove() & .pop()

if __name__ == '__main__':
    n = int(input())
    s = set(map(int, input().split()))
    N = int(input())
    for i in range(N):
        cmd = input().split()
        if cmd[0] == 'pop':
            s.pop()
        elif cmd[0] == 'remove':
            s.remove(int(cmd[1]))
        else: # cmd[0] == 'discard'
            s.discard(int(cmd[1]))
    print(sum(s))

# Set .union() Operation

if __name__ == '__main__':
    n = int(input())
    eng = set(map(int, input().split()))
    b = int(input())
    fre = set(map(int, input().split()))
    print(len(eng.union(fre)))

# Set .intersection() Operation

if __name__ == '__main__':
    n = int(input())
    eng = set(map(int, input().split()))
    b = int(input())
    fre = set(map(int, input().split()))
    print(len(eng.intersection(fre)))

# Set .difference() Operation

if __name__ == '__main__': 
    n = int(input())
    eng = set(map(int, input().split()))
    b = int(input())
    fre = set(map(int, input().split()))
    print(len(eng.difference(fre)))

# Set .symmetric_difference() Operation

if __name__ == '__main__': 
    n = int(input())
    eng = set(map(int, input().split()))
    b = int(input())
    fre = set(map(int, input().split()))
    print(len(eng.symmetric_difference(fre)))

# Set Mutations

if __name__ == '__main__': 
    n = int(input())
    A = set(map(int, input().split()))
    N = int(input())
    for i in range(N):
        opr = input().split()[0]
        B = set(map(int, input().split()))
        if opr == 'update':
            A.update(B)
        elif opr == 'intersection_update':
            A.intersection_update(B)
        elif opr == 'difference_update':
            A.difference_update(B)
        else: # opr == 'symmetric_difference_update'
            A.symmetric_difference_update(B)
    print(sum(A))

# The Capitain's Room

if __name__ == '__main__':
    K = int(input())
    rooms = list(map(int, input().split()))
    rooms.sort()
    loop_end = True
    for i in range(0, len(rooms)-K+1, K):
        if rooms[i] != rooms[i+K-1]:
            print(rooms[i])
            loop_end = False
            break
    if loop_end:
        print(rooms[-1])

# Check Subset

if __name__ == '__main__':
    T = int(input())
    for i in range(T):
        len_A = int(input())
        A = set(map(int, input().split()))
        len_B = int(input())
        B = set(map(int, input().split()))
        print( len(B.difference(A)) == (len_B - len_A) )

# Check Strict Superset

if __name__ == '__main__': 
    A = set(map(int, input().split()))
    N = int(input())
    strict_subset = True
    i = 0
    while(strict_subset and i<N):
        B = set(map(int, input().split()))
        difference_len = len(A.difference(B))
        if (difference_len != len(A) - len(B)) or difference_len == 0:
            strict_subset = False
        i += 1
    print(strict_subset)

# collections.Counter()

from collections import Counter

if __name__ == '__main__': 
    X = int(input())
    sizes = Counter(input().split())
    N = int(input())
    total = 0
    for i in range(N):
        size, price = input().split()
        if sizes[size] > 0:
            total += int(price)
            sizes[size] -= 1
    print(total)

# DefaultDict Tutorial

from collections import defaultdict

if __name__ == '__main__':
    n, m = tuple(map(int, input().split()))
    a = defaultdict(list)
    for i in range(1, n+1):
        a[input()].append(i)
    for i in range(m):
        b = input()
        if len(a[b]) != 0:
            print(*a[b])
        else:
            print(-1)

# Collections.namedtuple()

from collections import namedtuple

if __name__ == '__main__':
    n = int(input())
    columns = input().split()
    Row = namedtuple('Row', columns)
    marks_sum = 0
    for i in range(n):
        r = Row._make(input().split())
        marks_sum += int(r.MARKS)
    print(round(marks_sum/n, 2))

# Collections.OrderedDict()

from collections import OrderedDict

if __name__ == '__main__':
    n = int(input())
    dic = OrderedDict()
    for i in range(n):
        line = input().split()
        key = ' '.join(line[:-1])
        if dic.get(key) == None:
            dic[key] = int(line[-1])
        else:
            dic[key] += int(line[-1])
    for key in dic:
        print('{} {}'.format(key, dic[key])) 

# Word Order

from collections import OrderedDict

if __name__ == '__main__':
    n = int(input())
    dic = OrderedDict()
    for i in range(n):
        word = input()
        if dic.get(word) == None:
            dic[word] = 1
        else:
            dic[word] += 1
    print(len(list(dic.keys())))
    print(*[dic[key] for key in dic])

# Collections.deque()

from collections import deque

if __name__ == '__main__':
    n = int(input())
    q = deque()
    for i in range(n):
        op = input().split()
        if op[0] == 'append':
            q.append(int(op[1]))
        elif op[0] == 'appendleft':
            q.appendleft(int(op[1]))
        elif op[0] == 'pop':
            q.pop()
        else:
            q.popleft()
    print(*q)

# Company Logo

import math
import os
import random
import re
import sys
from collections import Counter

if __name__ == '__main__':
    s = input()
    dic = Counter(s)
    most_common = sorted(dic.items(), key=lambda x: (-x[1], x[0]))
    for c in most_common[:3]:
        print('{} {}'.format(*c))

# Piling Up!

from collections import deque

if __name__ == '__main__':
    t = int(input())
    for _ in range(t):
        n = int(input())
        q = deque(list(map(int, input().split())))
        stack = [2**31]
        possible = True
        while len(q) > 1 and possible:
            left_side = q.popleft()
            right_side = q.pop()
            if left_side >= right_side and left_side <= stack[-1]:
                stack.append(left_side)
                q.append(right_side)
            elif right_side > left_side and right_side <= stack[-1]:
                stack.append(right_side)
                q.appendleft(left_side)
            else:
                possible = False
        if len(q) > 0 and q.pop() > stack[-1]:
            possible = False
        print('Yes') if possible else print('No')

# Calendar Module

import calendar

if __name__ == '__main__':
    date = list(map(int, input().split()))
    weekdays = ('MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY')
    print(weekdays[ calendar.weekday(date[2], date[0], date[1]) ])

# Time Delta

import math
import os
import random
import re
import sys
from datetime import datetime

def time_delta(t1, t2):
    t1_obj = datetime.strptime(t1, '%a %d %b %Y %H:%M:%S %z')
    t2_obj = datetime.strptime(t2, '%a %d %b %Y %H:%M:%S %z')
    return str(abs((t1_obj - t2_obj)).total_seconds())[:-2]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()

# Exceptions 

if __name__ == '__main__':
    T = int(input())
    for i in range(T):
        a, b = input().split()
        try:
            print(int(a)//int(b))
        except BaseException as e:
            print(f'Error Code: {e}')

# Zipped!

if __name__ == '__main__':
    N, X = tuple(map(int, input().split()))
    table = []
    for i in range(X):
        table.append(map(float, input().split()))
    for marks in zip(*table):
        print(round(sum(marks)/X, 1))

# Athlete Sort

import math
import os
import random
import re
import sys

if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
    arr_sorted = sorted(arr, key=lambda x: x[k])
    for i in range(len(arr_sorted)):
        print(*arr_sorted[i])

# ginortS

if __name__ == '__main__':
    s = input()
    lower_case = upper_case = ''
    numbers = ['']*2
    for c in s:
        if c >= 'a':
            lower_case += c
        elif c >= '0' and c <= '9':
            if int(c) % 2 == 1:
                numbers[0] += c
            else:
                numbers[1] += c
        else:
            upper_case += c
    lower_case = ''.join(sorted(lower_case))
    upper_case = ''.join(sorted(upper_case))
    numbers[0] = ''.join(sorted(numbers[0]))
    numbers[1] = ''.join(sorted(numbers[1]))
    print(lower_case + upper_case + numbers[0] + numbers[1])

# Map and Lambda Function

cube = lambda x: x**3

def get_fibonacci_n(n):
    if n <= 1:
        return n
    else: 
        return get_fibonacci_n(n-2) + get_fibonacci_n(n-1)

def fibonacci(n):
    l = []
    for i in range(n):
        l.append(get_fibonacci_n(i))
    return l

if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))

# Detecting Floating Point Number

import re

if __name__ == '__main__':
    T = int(input())
    for i in range(T):
        string = input()
        pattern = re.compile('^([+-]?[0-9]*[.]{1}[0-9]+)$')
        print(bool(pattern.match(string)))

# Re.split() 

regex_pattern = r",|\."	# Do not delete 'r'.

import re
print("\n".join(re.split(regex_pattern, input())))

# Group(), Groups() & Groupdict()

import re

if __name__ == '__main__':
    string = input()
    match = re.search(r'([a-zA-Z0-9])(\1{1,})', string)
    print(match.group(1)) if match else print(-1)

# Re.findall() & Re.finditer()

import re

if __name__ == '__main__':
    string = input()
    pattern = r'(?=([^AEIOUaeiou0-9\s+-][AEIOUaeiou]{2,}[^AEIOUaeiou0-9\s+-]))'
    iterator = re.finditer(pattern, string)
    substrs = [match.group(1) for match in iterator]
    if substrs:
        for s in substrs:
            print(s[1:-1])
    else:
        print(-1)

# Re.start() & Re.end()

import re

if __name__ == '__main__':
    string = input()
    k = r'(?=(%s))' % input()
    iterator = re.finditer(k, string)
    iterator_empty = True
    for match in iterator:
        print(f'({match.start()}, {match.end() + len(match.group(1)) - 1})')
        iterator_empty = False
    if iterator_empty:
        print('(-1, -1)')

# Regex Substitution

import re

def replace(match):
    string = match.group(1)
    if string == '&&':
        return 'and'
    else: 
        return 'or'

if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        pattern = r'(?<= )(&&)(?= )|(?<= )(\|\|)(?= )'
        print(re.sub(pattern, replace, input()))

# Validating Roman Numerals

regex_pattern = r'^M{0,3}(CM|CD|D)?(XC|C{1,3})?(XL|L)?(|X{0,3})((IV|IX)?|V?I{0,3}?)$'	# Do not delete 'r'.

import re
print(str(bool(re.match(regex_pattern, input()))))

# Validating phone numbers

import re

if __name__ == '__main__': 
    n = int(input())
    for i in range(n):
        string = input()
        pattern = r'^([7-9]{1}[0-9]{9})$'
        match = re.match(pattern, string)
        if bool(match) == True:
            print('YES')
        else:
            print('NO')

# Validating and Parsing Email Addresses

import email.utils
import re

if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        string = email.utils.parseaddr(input())
        pattern = r'^[A-Za-z][\w\-.]*@[A-Za-z]+\.[A-Za-z]{1,3}$'
        match = re.match(pattern, string[1])
        if bool(match):
            print(email.utils.formataddr(string))

# Hex Color Code

import re

if __name__ == '__main__':
    n = int(input())
    css_code = ''
    for i in range(n):
        css_code += input()
    pattern = r'(#[\dA-Fa-f]{3,6})(?=;|,|\))'
    list_hex = re.findall(pattern, css_code)
    for match in list_hex:
        print(match)

# HTML Parser - Part 1

from html.parser import HTMLParser

def print_attrs(attrs):
    if len(attrs) > 0:
            for attr in attrs:
                print(f'-> {attr[0]} > ', end='')
                print(f'{attr[1]}') if len(attr)>1 else print('None')

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(f'Start : {tag}')
        print_attrs(attrs)

    def handle_endtag(self, tag):
        print(f'End   : {tag}')

    def handle_startendtag(self, tag, attrs):
        print(f'Empty : {tag}')
        print_attrs(attrs)

    def handle_comment(self, data):
        return ''

if __name__ == '__main__':
    n = int(input())
    html_code = ''
    for i in range(n):
        html_code += input()
    parser = MyHTMLParser()
    parser.feed(html_code)

# HTML Parser - Part 2 

from html.parser import HTMLParser

def check_data(data):
    return data != '\n'

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if data.find('\n') == -1: # single-line comment
            print('>>> Single-line Comment')
        else:
            print('>>> Multi-line Comment')
        if check_data(data):
            print(data) 

    def handle_data(self, data):
        if check_data(data):
            print(f'>>> Data\n{data}') 

html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()

# Detecting HTML Tags, Attributes and Attribute Values 

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for attr in attrs:
            print(f'-> {attr[0]} > {attr[1]}')
    
    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_comment(self, data):
        return ''

if __name__ == '__main__':
    n = int(input())
    html_code = ''
    for i in range(n):
        html_code += input()
    parser = MyHTMLParser()
    parser.feed(html_code)

# Validating UID

import re 

if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        uid = input()
        # pattern = '^((?=.*\d{3,})(?=.*[A-Z]{2,})([A-Za-z\d])([A-Za-z\d]^\1)){10}$'
        pattern_format = r'^[A-Za-z\d]{10}$'
        pattern_alpha = r'[A-Z]'
        pattern_num = r'\d'
        pattern_repetition = r'(.).*\1'
        valid = True
        if not bool(re.match(pattern_format, uid)):
            valid = False
        if valid and len(re.findall(pattern_alpha, uid)) < 2:
            valid = False
        if valid and len(re.findall(pattern_num, uid)) < 3:
            valid = False
        if valid and re.search(pattern_repetition, uid) != None:
            valid = False
        print('Valid') if valid else print('Invalid')

# Validating Credit Card Numbers

import re

if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        string = input()
        valid = True
        pattern_format = r'^(([4-6]\d{3})([-]?\d{4}){3})$'
        pattern_repetition = r'(.)([\-]?\1){3,}'
        if not bool(re.match(pattern_format, string)):
            valid = False
        if valid and re.search(pattern_repetition, string) != None:
            valid = False
        
        if valid:
            print('Valid')
        else:
            print('Invalid')

# Validating Postal Codes

regex_integer_in_range = r"^[1-9][0-9]{5}$"	# Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"(?=(.).\1)(.)"	# Do not delete 'r'.

import re
P = input()
print (bool(re.match(regex_integer_in_range, P)) and len(re.findall(regex_alternating_repetitive_digit_pair, P)) < 2)

# Matrix Script

import math
import os
import random
import re
import sys

first_multiple_input = input().rstrip().split()
n = int(first_multiple_input[0])
m = int(first_multiple_input[1])
matrix = ['']*m
pattern = r'(?<=[A-Za-z\d])[^A-Za-z\d]+(?=[A-Za-z\d])' 
decoded_text = ''
for i in range(n):
    matrix_item = input()
    for j in range(m):
        matrix[j] += matrix_item[j]

text = ''.join(matrix)
decoded_text += re.sub(pattern, ' ', text)
print(decoded_text)

# XML 1 - Find the Score

import sys
import xml.etree.ElementTree as etree

def get_attr_number(root):
    score = 0
    for child in root.iter():
        score += len(child.attrib)
    return score

if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))

# XML2 - Find the Maximum Depth

import xml.etree.ElementTree as etree

def depth_routine(elem):
    if len(list(elem)) == 0:
        return 0
    children = []
    for child in list(elem):
        children.append(depth_routine(child))
    max_depth = max(children)
    return max_depth + 1 

maxdepth = 0
def depth(elem, level):
    global maxdepth
    maxdepth = depth_routine(elem)
    
if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml =  xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)

# Standardize Mobile Number Using Decorators

def wrapper(f):
    def fun(l):
        f(['+91 ' + num[-10:-5] + ' ' + num[-5:] for num in l])
    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l) 

# Decorators 2 - Name Directory

import operator

def person_lister(f):
    def inner(people):
        people = [[person[0], person[1], int(person[2]), person[3]] for person in people]
        people.sort(key = operator.itemgetter(2))
        return [f(person) for person in people]
    return inner

@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]

if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')

# Arrays

import numpy

def arrays(arr):
    return numpy.array(arr[::-1], float)

arr = input().strip().split(' ')
result = arrays(arr)
print(result)

# Shape and Reshape

import numpy

if __name__ == '__main__':
    l = input().split()
    print(numpy.reshape(numpy.array(l, int), (3,3)))

# Transpose and Flatten

import numpy

if __name__ == '__main__': 
    n, m = tuple(map(int, input().split()))
    matrix = []
    for i in range(n): 
        matrix.append(input().split())
    my_array = numpy.array(matrix, int)
    print(my_array.transpose())
    print(my_array.flatten())

# Concatenate

import numpy

if __name__ == '__main__':
    n, m, p = tuple(map(int, input().split()))
    matrix_1 = [list(map(int, input().split())) for i in range(n)]
    matrix_2 = [list(map(int, input().split())) for i in range(m)]
    print(numpy.concatenate((matrix_1, matrix_2), axis=0))


# Zeros and Ones

import numpy

if __name__ == '__main__':
    n = list(map(int, input().split()))
    print(numpy.zeros(n, int))
    print(numpy.ones(n, int))
    
# Eye and Identity

import numpy

if __name__ == '__main__': 
    n, m = tuple(map(int, input().split()))
    numpy.set_printoptions(sign=' ')
    print(numpy.eye(n, m))

# Array Mathematics

import numpy

if __name__ == '__main__': 
    n, m = tuple(map(int, input().split()))
    l = []
    for i in range(n):
        l.append(input().split())
    a = numpy.array(l, int)
    l = []
    for i in range(n):
        l.append(input().split())   
    b = numpy.array(l, int)
    print(a+b)
    print(a-b)
    print(a*b)
    print(a//b)
    print(a%b)
    print(a**b)

# Floor, Ceil and Rint

import numpy

if __name__ == '__main__':
    n = numpy.array(input().split(), float)
    numpy.set_printoptions(sign=' ')
    print(numpy.floor(n))
    print(numpy.ceil(n))
    print(numpy.rint(n))

# Sum and Prod

import numpy

if __name__ == '__main__':
    n, m = tuple(map(int, input().split()))
    l = [input().split() for i in range(n)]
    my_array = numpy.array(l, int)
    my_array = numpy.sum(my_array, axis=0)
    print(numpy.prod(my_array))

# Min and Max

import numpy

if __name__ == '__main__':
    n, m = tuple(map(int, input().split()))
    l = [input().split() for i in range(n)]
    my_array = numpy.array(l, int)
    my_array = numpy.min(my_array, axis=1)
    print(numpy.max(my_array))

# Mean, Var and Std

import numpy

if __name__ == '__main__':
    n, m = tuple(map(int, input().split()))
    l = [input().split() for i in range(n)]
    my_array = numpy.array(l, int)
    mean = numpy.mean(my_array, axis=1)
    var = numpy.var(my_array, axis=0)
    std = numpy.std(my_array, )
    numpy.set_printoptions(sign=' ')
    print(mean)
    print(var)
    print(round(std, 12))

# Dot and Cross

import numpy

if __name__ == '__main__':
    n = int(input())
    a_lst = [input().split() for i in range(n)]
    b_lst = [input().split() for i in range(n)]
    a = numpy.array(a_lst, int)
    b = numpy.array(b_lst, int)
    print(numpy.dot(a, b, ))

# Inner and Outer

import numpy

if __name__ == '__main__':
    a = numpy.array(input().split(), int)
    b = numpy.array(input().split(), int)
    print(numpy.inner(a,b))
    print(numpy.outer(a,b))

# Polynomials

import numpy

if __name__ == '__main__':
    p = numpy.array(input().split(), float)
    x = float(input())
    print(numpy.polyval(p, x))

# Linear Algebra

import numpy

if __name__ == '__main__': 
    n = int(input())
    l = [input().split() for i in range(n)]
    a = numpy.array(l, float)
    print(round(numpy.linalg.det(a), 2))

# Birthday Cake Candles

import math
import os
import random
import re
import sys

def birthdayCakeCandles(candles):
    max_candle = max(candles)
    return candles.count(max_candle)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    candles_count = int(input().strip())
    candles = list(map(int, input().rstrip().split()))
    result = birthdayCakeCandles(candles)
    fptr.write(str(result) + '\n')
    fptr.close()

# Number Line Jump

import math
import os
import random
import re
import sys

def kangaroo(x1, v1, x2, v2):
    t = (x2 - x1) / (v1 - v2) if (v1 - v2) != 0 else -1
    return 'YES' if t >= 0 and t == int(t) else 'NO'

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    x1V1X2V2 = input().split()
    x1 = int(x1V1X2V2[0])
    v1 = int(x1V1X2V2[1])
    x2 = int(x1V1X2V2[2])
    v2 = int(x1V1X2V2[3])
    result = kangaroo(x1, v1, x2, v2)
    fptr.write(result + '\n')
    fptr.close()

# Viral Advertising

import math
import os
import random
import re
import sys

def day_liked(n):
    if n == 1:
        return 2
    return math.floor(day_liked(n-1) * 3/2)

def viralAdvertising(n):
    s = 0
    for i in range(1, n+1):
        s += day_liked(i)
    return s

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input())
    result = viralAdvertising(n)
    fptr.write(str(result) + '\n')
    fptr.close()

# Recursive Digit Sum

import math
import os
import random
import re
import sys

def superDigit(n, k):
    number = list(n)
    if len(number) == 1:
        return int(number[0])
    digits = list(map(int, number))
    return superDigit(str(sum(digits) * k), 1)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    nk = input().split()
    n = nk[0]
    k = int(nk[1])
    result = superDigit(n, k)
    fptr.write(str(result) + '\n')
    fptr.close()

# Insertion Sort - Part 1

import math
import os
import random
import re
import sys

def insertionSort1(n, arr):
    last_n = arr[-1]
    i = n - 2
    while i >= 0  and arr[i] > last_n:
        arr[i+1] = arr[i]
        print(*arr)
        i -= 1
    arr[i+1] = last_n
    print(*arr)

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().rstrip().split()))
    insertionSort1(n, arr)

# Insertion Sort - Part 2

import math
import os
import random
import re
import sys

def insertionSort2(n, arr):
    for i in range(1, n):
        selected = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > selected:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = selected
        print(*arr)

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().rstrip().split()))
    insertionSort2(n, arr)

