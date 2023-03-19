# SQL & DB Interview Questions & Answers for Data Scientists # 

## Questions ##
* [Q1: What are joins in SQL and discuss its types?](https://github.com/youssefHosni/Data-Science-Interview-Questions-Answers/blob/main/SQL%20%26%20DB%20Interview%20Questions%20%26%20Answers%20for%20Data%20Scientists.md#q1-what-are-joins-in-sql-and-discuss-its-types)
* [Q2: Define the primary, foreign, and unique keys and the differences between them?](https://github.com/youssefHosni/Data-Science-Interview-Questions-Answers/blob/main/SQL%20%26%20DB%20Interview%20Questions%20%26%20Answers%20for%20Data%20Scientists.md#q2-define-the-primary-foreign-and-unique-keys-and-the-differences-between-them)
* [Q3: What is the difference between BETWEEN and IN operators in SQL?](https://github.com/youssefHosni/Data-Science-Interview-Questions-Answers/blob/main/SQL%20%26%20DB%20Interview%20Questions%20%26%20Answers%20for%20Data%20Scientists.md#q3-what-is-the-difference-between-between-and-in-operators-in-sql)
* [Q4: Assume you have the given table below which contains information on user logins. Write a query to obtain the number of reactivated users (Users who did not log in the previous month and then logged in the current month)](https://github.com/youssefHosni/Data-Science-Interview-Questions-Answers/blob/main/SQL%20%26%20DB%20Interview%20Questions%20%26%20Answers%20for%20Data%20Scientists.md#q4-assume-you-have-the-given-table-below-which-contains-information-on-user-logins-write-a-query-to-obtain-the-number-of-reactivated-users-users-who-did-not-log-in-the-previous-month-and-then-logged-in-the-current-month)
![Alt_text](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Figures/Screenshot%202022-07-31%20201033.png)
* [Q5: Describe the advantages and disadvantages of relational database vs NoSQL databases](https://github.com/youssefHosni/Data-Science-Interview-Questions-Answers/blob/main/SQL%20%26%20DB%20Interview%20Questions%20%26%20Answers%20for%20Data%20Scientists.md#q5-describe-the-advantages-and-disadvantages-of-relational-database-vs-nosql-databases)
* [Q6: Assume you are given the table below on user transactions. Write a query to obtain the third transaction of every user](https://github.com/youssefHosni/Data-Science-Interview-Questions-Answers/blob/main/SQL%20%26%20DB%20Interview%20Questions%20%26%20Answers%20for%20Data%20Scientists.md#q6-assume-you-are-given-the-table-below-on-user-transactions-write-a-query-to-obtain-the-third-transaction-of-every-user)
![1661352126442](https://user-images.githubusercontent.com/72076328/186479577-da475779-b4de-45ef-b1ec-79ca5df0dad5.png)
* [Q7: What do you understand by Self Join? Explain using an example](https://github.com/youssefHosni/Data-Science-Interview-Questions-Answers/blob/main/SQL%20%26%20DB%20Interview%20Questions%20%26%20Answers%20for%20Data%20Scientists.md#q7-what-do-you-understand-by-self-join-explain-using-an-example)
* [Q8: Write an SQL query to join 3 tables](https://github.com/youssefHosni/Data-Science-Interview-Questions-Answers/blob/main/SQL%20%26%20DB%20Interview%20Questions%20%26%20Answers%20for%20Data%20Scientists.md#q8-write-an-sql-query-to-join-3-tables)
* [Q9: Write a SQL query to get the third-highest salary of an employee from employee_table and arrange them in descending order.](https://github.com/youssefHosni/Data-Science-Interview-Questions-Answers/blob/main/SQL%20%26%20DB%20Interview%20Questions%20%26%20Answers%20for%20Data%20Scientists.md#q9-write-a-sql-query-to-get-the-third-highest-salary-of-an-employee-from-employee_table-and-arrange-them-in-descending-order)
* [Q10: What is the difference between temporary tables and common table expressions?](https://github.com/soopertramp/Data-Science-Interview-Questions-Answers/blob/main/SQL%20&%20DB%20Interview%20Questions%20&%20Answers%20for%20Data%20Scientists.md#q10-what-is-the-difference-between-temporary-tables-and-common-table-expressions)
* [Q11: Why use Right Join When Left Join can suffice the requirement?](https://github.com/soopertramp/Data-Science-Interview-Questions-Answers/blob/main/SQL%20&%20DB%20Interview%20Questions%20&%20Answers%20for%20Data%20Scientists.md#q11-why-use-right-join-when-left-join-can-suffice-the-requirement)
* [Q12: Why Rank skips sequence?](https://github.com/soopertramp/Data-Science-Interview-Questions-Answers/blob/main/SQL%20&%20DB%20Interview%20Questions%20&%20Answers%20for%20Data%20Scientists.md#q12-why-rank-skips-sequence)
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
### Q5: Describe the advantages and disadvantages of relational database vs NoSQL databases ###

Answer:

**Advantages of Relational Databases:** Ensure data integrity through a defined schema and ACID properties.  Easy to get started with and use for small-scale applications. Lends itself well to vertical scaling.  Uses an almost standard query language, making learning or switching between types of relational databases easy.

**Advantages of NoSQL Databases:** Offers more flexibility in data format and representations, which makes working with Unstructured or semistructured data easier.  Hence, useful when still the data schema or adding new features/functionality rapidly like in a startup environment to scale with horizontal scaling.  Lends itself better to applications that need to be highly available. 

**Disadvantages of Relational Databases:** Data schema needs to be known in advance.  Ale schemas is possible, but frequent changes to the schema for large tables can cause performance issues.  Horizontal scaling is relatively difficult, leading to eventual performance bottlenecks

**Disadvantages of NoSQL Databases:** As outlined by the BASE framework, weaker guarantees of data correctness are made due to the soft-state and eventual consistency property.  Managing consistency can also be difficult due to the lack of a predefined schema that's strictly adhered to. Depending on the type of NoSQL database, it can be challenging for the database to handle its types of complex queries or access patterns.


![Alt_text](https://github.com/youssefHosni/Data-Science-Interview-Questions/blob/main/Figures/ezgif.com-gif-maker.jpg)


### Q6: Assume you are given the table below on user transactions. Write a query to obtain the third transaction of every user ###

![1661352126442](https://user-images.githubusercontent.com/72076328/186479648-5dcbf0a3-46e3-46b4-99ab-94feeace3dca.png)

Answer:
First, we obtain the transaction numbers for each user. We can do this by using the ROW_NUMBER window function, where we PARTITION by the user_id and ORDER by the transaction_date fields, calling the resulting field a transaction number. From there, we can simply take all transactions having a transaction number equal to 3.
![1661352088335](https://user-images.githubusercontent.com/72076328/186479695-5d2b7f36-5703-489d-87e3-6bad6ee1a9b7.jpg)

### Q7: What do you understand by Self Join? Explain using an example ###

Answer:

Self-join is as its name implies, joining a table to itself on a database, this process may come in handy in a number of cases, such as:

1- comparing the table's rows to themselves:

It's like we have two copies of the same table and join them together on a given condition to reach the required output query.

Ex. If we have a store database with a client's data table holding a bunch of demographics, we could self-join the client's table to get clients who are located in the same city/made a purchase on the same day/etc.

2- querying a table that has hierarchical data:

Meaning, the table has a primary key that has a one-to-many relationship with another foreign key inside the same table, in other words, the table has data that refers to the same table. We could use self-join in order to have a clear look at the data by matching its keys.

Ex. The organizational structure of a company may contain an employee table that has an employee id and his manager id (who is also an employee, hence has an employee id too) in the same table. Using self-join on this table would allow us to reference every employee directly to his manager.

P.S. we would need to take care of duplicates that may occur and consider them in the conditions.

### Q8: Write an SQL query to join 3 tables ###
![1668274347333](https://user-images.githubusercontent.com/72076328/201538710-264494b8-62e7-4e36-8487-be449c1b441a.jpg)


### Q9: Write a SQL query to get the third-highest salary of an employee from employee_table and arrange them in descending order. ###

Answer:

### Q10: What is the difference between temporary tables and common table expressions? ###

Answer:

𝗧𝗲𝗺𝗽𝗼𝗿𝗮𝗿𝘆 𝘁𝗮𝗯𝗹𝗲𝘀 and 𝗖𝗧𝗘s are both used to store intermediate results in MySQL, but there are some key differences between the two:

𝗗𝗲𝗳𝗶𝗻𝗶𝘁𝗶𝗼𝗻: A temporary table is a physical table that is created in the database and persists until it is explicitly dropped or the session ends. A CTE is a virtual table that is defined only within the scope of a single SQL statement.

𝗦𝘁𝗼𝗿𝗮𝗴𝗲: Temporary tables are stored in the database and occupy physical disk space. CTEs are not stored on disk and exist only in memory for the duration of the query.

𝗔𝗰𝗰𝗲𝘀𝘀: Temporary tables can be accessed from any session that has the appropriate privileges. CTEs are only accessible within the scope of the query in which they are defined.

𝗟𝗶𝗳𝗲𝘀𝗽𝗮𝗻: Temporary tables persist until they are explicitly dropped or the session ends. CTEs are only available for the duration of the query in which they are defined and are then discarded.

𝗦𝘆𝗻𝘁𝗮𝘅: Temporary tables are created using the CREATE TEMPORARY TABLE statement, while CTEs are defined using the WITH clause.

𝗣𝘂𝗿𝗽𝗼𝘀𝗲: Temporary tables are typically used to store intermediate results that will be used in multiple queries, while CTEs are used to simplify complex queries by breaking them down into smaller, more manageable parts.

In summary, temporary tables are physical tables that persist in the database and can be accessed from any session, while CTEs are virtual tables that exist only within the scope of a single query and are discarded once the query is complete. Both temporary tables and CTEs can be useful tools for simplifying complex queries and storing intermediate results.

### Q11: Why use Right Join When Left Join can suffice the requirement? ###

Answer:
In MySQL, the 𝗥𝗜𝗚𝗛𝗧 𝗝𝗢𝗜𝗡 𝗮𝗻𝗱 𝗟𝗘𝗙𝗧 𝗝𝗢𝗜𝗡 are used to retrieve data from multiple tables by joining them based on a specified condition.

Generally, the 𝗟𝗘𝗙𝗧 𝗝𝗢𝗜𝗡 is used more frequently than the 𝗥𝗜𝗚𝗛𝗧 𝗝𝗢𝗜𝗡 because it returns all the rows from the left table and matching rows from the right table, or NULL values if there is no match.

In most cases, a 𝗟𝗘𝗙𝗧 𝗝𝗢𝗜𝗡 is sufficient to meet the requirement of retrieving all the data from the left table and matching data from the right table.

However, there may be situations where using a 𝗥𝗜𝗚𝗛𝗧 𝗝𝗢𝗜𝗡 is more appropriate.

Here are a few examples:

𝟭. 𝗪𝗵𝗲𝗻 𝘁𝗵𝗲 𝗽𝗿𝗶𝗺𝗮𝗿𝘆 𝘁𝗮𝗯𝗹𝗲 𝗶𝘀 𝘁𝗵𝗲 𝗿𝗶𝗴𝗵𝘁 𝘁𝗮𝗯𝗹𝗲: If the right table contains the primary data that needs to be retrieved, and the left table contains supplementary data, a 𝗥𝗜𝗚𝗛𝗧 𝗝𝗢𝗜𝗡 can be used to retrieve all the data from the right table and matching data from the left table.

𝟮. 𝗪𝗵𝗲𝗻 𝘁𝗵𝗲 𝗾𝘂𝗲𝗿𝘆 𝗻𝗲𝗲𝗱𝘀 𝘁𝗼 𝗯𝗲 𝗼𝗽𝘁𝗶𝗺𝗶𝘇𝗲𝗱: In some cases, a 𝗥𝗜𝗚𝗛𝗧 𝗝𝗢𝗜𝗡 may be more efficient than a 𝗟𝗘𝗙𝗧 𝗝𝗢𝗜𝗡 because the database optimizer can choose the most efficient join order based on the query structure and the available indexes.

𝟯. 𝗪𝗵𝗲𝗻 𝘂𝘀𝗶𝗻𝗴 𝗼𝘂𝘁𝗲𝗿 𝗷𝗼𝗶𝗻𝘀: If the query requires an outer join, a 𝗥𝗜𝗚𝗛𝗧 𝗝𝗢𝗜𝗡 may be used to return all the rows from the right table, including those with no matching rows in the left table.
It's important to note that while a 𝗥𝗜𝗚𝗛𝗧 𝗝𝗢𝗜𝗡 can provide additional functionality in certain cases, it may also make the query more complex and difficult to read. In most cases, a 𝗟𝗘𝗙𝗧 𝗝𝗢𝗜𝗡 is the preferred method for joining tables in MySQL.

### Q12: Why Rank skips sequence? ###

Answers:
In MySQL, the rank function may skip a sequence of numbers when using the `DENSE_RANK()` function or the `RANK()` function, depending on the data and the query. The `DENSE_RANK()` function assigns a unique rank to each distinct value in a result set, whereas the `RANK()` function assigns the same rank to the duplicate values.

Here are some of the reasons why the rank function may skip a sequence in MySQL:

1. 𝗧𝗵𝗲 `𝗗𝗘𝗡𝗦𝗘_𝗥𝗔𝗡𝗞()` function skips ranks when there are ties. For example, if there are two rows with the same values in the ranking column, both will be assigned the same rank, and the next rank will be incremented by 1.

2. 𝗧𝗵𝗲 `𝗥𝗔𝗡𝗞()` function skips ranks when there are gaps between the duplicate values. For example, if there are three rows with the same values in the ranking column, and then the next row has a higher value, the `RANK()` function will skip over the fourth rank.

3. The query may have filtering or grouping clauses that affect the ranking. For example, if a query filters out some rows or groups them by a different column, the ranking may not be sequential.

It's important to note that the ranking function in MySQL behaves differently from the ranking function in other databases, so the same query may produce different results in different database systems.
