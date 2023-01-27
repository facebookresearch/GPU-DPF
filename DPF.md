# Private Information Retrieval with Distributed Point Functions

This note describes how **distributed point functions (DPF)** can be used to enable private accesses to a table stored on two non-colluding servers.

## Private Table Lookups
Suppose Meta servers hold a database (**T**) as shown below. The content of the table is known to Meta, but the table needs to be accessed with private indices.

<p align="center">
<img src="https://github.com/facebookresearch/GPU-DPF/blob/main/imgs/1.png" width="180">
</p>

Let’s say that a user wants to access the 4th entry (index 3) of the table, obtaining 4, but without Meta knowing which index he or she accessed. How can we do this?

A useful framework for understanding table lookups is to view them as a dot product. That is, a table lookup is essentially doing a dot product between the entire table and a one-hot vector containing a ‘1’ at the index of the item the user wants to look up, and ‘0’ everywhere else. 
The previous example of looking up the 4th entry of table T is shown below.

<p align="center">
<img src="https://github.com/facebookresearch/GPU-DPF/blob/main/imgs/2.png" width="300">
</p>

Performing the dot product and returning the result would give the correct answer, 4 – but it would not hide the user’s query!

To fix this, we can introduce a second non-colluding server that holds the same table T. 

Then, the user can generate a random vector R, and query one server to return the result of the dot product T\*(R+Q), and the other to return the result of the dot-product T\*R (assuming we are working within a finite field). 

The difference between the results of the two servers gives T\*Q, which is exactly the entry the user wanted! Furthermore, no information is leaked as R and (R+Q), when working over a finite field, individually reveal no information about the query! This technique is known as additive blinding, and R and R+Q are referred to as additive secret shares of Q.

<p align="center">
<img src="https://github.com/facebookresearch/GPU-DPF/blob/main/imgs/3.png" width="400">
</p>

This is great, but there’s a new issue: the size of the vectors that the user needs to send is big. 

Concretely, the number of entries in these vectors is proportional to the number of entries in the table. If we were privately looking up an element in a table with 10,000,000,000 entries, we would have to send over 1GB of data to the servers! Clearly, this dot-product method, as it is, is impractical for even moderately sized tables (e.g: tables with thousands or millions of entries would incur kilobytes or megabytes of communication cost).

This is where distributed point functions (DPF) come into play. 

A DPF is a way to compress the secret-shared one-hot vector significantly. Specifically, a **DPF is a cryptographic primitive that, when evaluated, yields secret shares of a vector that is zero everywhere except at a single location**. 

So, how much can DPFs compress secret sharings of one-hot vectors? Amazingly, if N is the number of entries in the table, a DPF can compress these shares to O(log(N)). This means, privately accessing an element in a 10,000,000,000 entry table no longer needs >1GB of communication, but only around 2-3 KB with a DPF! 

Below we’ll see how the distributed point function achieves this.

## How Distributed Point Functions Work

A **DPF** is a cryptographic primitive that allows us to compactly represent secret shares of a one-hot vector. 

Concretely, DPFs allow us to generate a pair of compact keys K1 and K2, which, when expanded, yield secret shares of a one hot vector with alpha as the target non-zero index (note, any scale beta, can be chosen for the output value for the target index):

<p align="center">
<img src="https://github.com/facebookresearch/GPU-DPF/blob/main/imgs/4.png" width="300">
</p>

Given this functionality, to perform a private table access, a user would construct keys K1 and K2, send each to the corresponding server, who would then expand the DPF with their key to obtain secret shares of the one-hot vector, then finally perform a dot product with their table to return the result. The user adds the two results from the servers to obtain the plaintext table entry.

Amazingly, the size of the keys K1 and K2 can be made to be O(log(N)) the size of the length of the table. The key idea is to view the server’s table as a 2-D grid rather than a 1-D array, where the number of rows and columns of the grid is O(sqrt(N)) the number of table elements.

<p align="center">
<img src="https://github.com/facebookresearch/GPU-DPF/blob/main/imgs/5.png" width="300">
</p>

Now, if the user’s target table index were say, 5, what the user can do is, assign server 1’s key K1 to be 3 random “nonces” (one for each column of the 2-D grid T’), and assign server 2’s key K2 to be the same “nonces”, except at the column containing the one-hot index. 
The two servers would then use these “nonces” as seeds for a random number generator (RNG) to generate values for each column.

<p align="center">
<img src="https://github.com/facebookresearch/GPU-DPF/blob/main/imgs/6.png" width="400">
</p>

As shown, for the green columns, which represent indices that are not the target, the DPF expansion would yield the exact same values, which is a secret share of 0! 

For the red column, which contains the target index, the DPF expansion would evaluate to something different, something random. Unfortunately, this entire column is different (but we only want a single particlar entry to be different) – so we need to do some error correction. 

To account for this difference, the user additionally sends two correction words to both servers, each the size of the length of a column. These two correction words have values that look random, but the difference between them is chosen to be the difference between the two RNGs (red column) plus a one at the row index for the target. 

During DPF expansion, the servers add the correction word indexed by the last bit of the nonce for that particular column. By doing this, we effectively “correct” the red column as the last bits for the differing nonces are chosen to be different. The corresponding “green” columns still evaluate to the same number, since they’ve added the exact same correction words.

<p align="center">
<img src="https://github.com/facebookresearch/GPU-DPF/blob/main/imgs/7.png" width="400">
</p>

As seen, these codewords allow us to “correct” the differences of the last column. With this, our expansion method now outputs secret shares of a one-hot vector. 

Security is maintained because all data sent to the servers (K1 individually, K2 individually, correction words), as well as the expanded output, look random. Assuming we use 128-bit nonces, it is computationally hard to brute force the seeds (e.g: try out all possible seeds) to do a pattern matching attack.

What is the required amount of communication for this scheme? As shown, if we have a table with N elements, then the length of the nonces (K1, K2) and codewords is sqrt(N). This is a large improvement over O(N) communication with the naive method.

But, as discussed earlier, we can do even better. We can reduce the communication down to log(N).

The critical observation is that, the nonces (e.g: {3, 5, 9} and {3, 5, 2}) can themselves be represented as a DPF, as they are the same at all places except one. We can recursively construct a DPF to generate them. Concretely, instead of viewing the table as a sqrt(N)-by-sqrt(N) grid, we view it as a [2-by-N/2] grid, and recursively construct DPFs for the N/2 nonces. This allows us to reduce communication to log(N) because the number of times we recurse is log(N), and we have only two correction words per level. In practice, using the log(N) scheme, a private query to a table with ~1,000,000 entries takes around 1 KB of communication.

## Conclusion
A DPF is a cryptographic primitive that, when evaluated, yields secret shares of a vector that is zero everywhere except at a single location. 

DPFs can be used to enable private table accesses to two untrusted non-colluding servers that share a table. They derive their security from other cryptographic primitives such as random number generators and PRFs. 

DPFs have applications to a wide range of cryptographic applications and is an important cryptographic tool for privacy preserving computation.
