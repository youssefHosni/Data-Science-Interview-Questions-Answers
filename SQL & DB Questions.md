# SQL & DB Questions # 

## Questions ##
* [Q1: What are joins in SQL and discuss its types?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/SQL%20&%20DB%20Questions.md#:~:text=Questions%20%26%20Answers-,Q1%3A%20What%20are%20joins%20in%20SQL%20and%20discuss%20its%20types%3F,-A%20JOIN%20clause)
* [Q2: Define the primary, foreign, and unique keys and the differences between them?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/SQL%20&%20DB%20Questions.md#:~:text=hand%20side%20table.-,Q2%3A%20Define%20the%20primary%2C%20foreign%2C%20and%20unique%20keys%20and%20the%20differences%20between%20them%3F,-Primary%20key%3A)
* [Q3: What is the difference between BETWEEN and IN operators in SQL?](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/SQL%20&%20DB%20Questions.md#:~:text=one%20Unique%20key-,Q3%3A%20What%20is%20the%20difference%20between%20BETWEEN%20and%20IN%20operators%20in%20SQL%3F,-Footer)
* [Q4: Assume you have the given table below which contains information on user logins. Write a query to obtain the number of reactivated users (Users who did not log in the previous month and then logged in the current month)](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/SQL%20&%20DB%20Questions.md#:~:text=for%20Numerical%20Variables-,Q4%3A%20Assume%20you%20have%20the%20given%20table%20below%20which%20contains%20information%20on%20user%20logins.%20Write%20a%20query%20to%20obtain%20the%20number%20of%20reactivated%20users%20(Users%20who%20did%20not%20log%20in%20the%20previous%20month%20and%20then%20logged%20in%20the%20current%20month),-Answer%3A)
![Alt_text](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Figures/Screenshot%202022-07-31%20201033.png)
* [Q5:  Q5:Describe the advantages and disadvantages of relational database vs NoSQL databases](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/SQL%20&%20DB%20Questions.md#:~:text=INTERVAL%20%271%20month%27%0A)-,Q5%3ADescribe%20the%20advantages%20and%20disadvantages%20of%20relational%20database%20vs%20NoSQL%20databases,-Footer)
---------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Questions & Answers ##

### Q1: What are joins in SQL and discuss its types? ###
A JOIN clause is used to combine rows from two or more tables, based on a related column between them. It is used to merge two tables or retrieve data from there. There are 4 types of joins: inner join left join, right join, and full join.

* Inner join: Inner Join in SQL is the most common type of join. It is used to return all the rows from multiple tables where the join condition is satisfied. 
* Left Join:  Left Join in SQL is used to return all the rows from the left table but only the matching rows from the right table where the join condition is fulfilled.
* Right Join: Right Join in SQL is used to return all the rows from the right table but only the matching rows from the left table where the join condition is fulfilled.
* Full Join: Full join returns all the records when there is a match in any of the tables. Therefore, it returns all the rows from the left-hand side table and all the rows from the right-hand side table.
![alt text](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Figures/Joins%20in%20SQL.png)

### Q2: Define the primary, foreign, and unique keys and the differences between them? ###

**Primary key:** Is a key that is used to uniquely identify each row or record in the table, it can be a single column or composite pk that contains more than one column

* The primary key doesn't accept null or repeated values
* The purpose of the primary key is to keep the Entity's integrity
* There is only one PK in each table
* Every row must have a unique primary key

**Foreign key:** Is a key that is used to identify, show or describe the relationship between tuples of two tables. It acts as a cross-reference between tables because it references the primary key of another table, thereby establishing a link between them.

* The purpose of the foreign key is to keep data integrity
* It can contain null values or primary key values

**Unique key:** It's a key that can identify each row in the table as the primary key but it can contain one null value

* Every table can have more than one Unique key

### Q3: What is the difference between BETWEEN and IN operators in SQL? ###
Answer:

The SQL **BETWEEN** operator selects values within a given range. It is inclusive of both the ranges, begin and end values are included.  The values can be text, date, numbers, or other

For example, select * from tablename where price BETWEEN 10 and 100;

The **IN** operator is used to select rows in which a certain value exists in a given field. It is used with the WHERE clause to match values in a list.

For example, select COLUMN from tablename where 'USA' in (country);

IN is mainly best for categorical variables(it can be used with Numerical as well) whereas Between is for Numerical Variables
![alt_text](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Figures/Betweem%26IN.png)

### Q4: Assume you have the given table below which contains information on user logins. Write a query to obtain the number of reactivated users (Users who did not log in the previous month and then logged in the current month) ###
![Alt_text](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Figures/Screenshot%202022-07-31%20201033.png)

Answer: 
First, we look at all the users who did not log in during the previous month. To obtain the last month's data, we subtract an 𝐈𝐍𝐓𝐄𝐑𝐕𝐀𝐋 of 1 month from the current month's login date. Then, we use 𝐖𝐇𝐄𝐑𝐄 𝐄𝐗𝐈𝐒𝐓𝐒 against the previous month's interval to check whether there was login in the previous month. Finally, we 𝗖𝗢𝗨𝗡𝗧 the number of users satisfying this condition.

```
SELECT 
    DATE_TRUNC('month', current_month.login_date) AS current_month,
    COUNT(*) AS num_reactivated_users 
FROM 
    user_logins current_month
WHERE
    NOT EXISTS (
      SELECT 
        *
      FROM 
        user_logins last_month
      WHERE
        DATE_TRUNC('month', last_month.login_date) BETWEEN DATE_TRUNC('month', current_month.login_date) AND DATE_TRUNC('month', current_month.login_date) - INTERVAL '1 month'
)
```
### Q5:Describe the advantages and disadvantages of relational database vs NoSQL databases ###
