package io.dhiraa

case class User(name: String, age: Int)

//scalac -Xprint:parse src/main/scala/io/dhiraa/SimpleForComprehension.scala

/**
  * A Scala for comprehension can contain the following three types of expressions:
  * - Generators
  * - Filters
  * - Definitions
  */

abstract class CustomClass[A] {
  def map[B](f: A => B): CustomClass[B]
  def flatMap[B](f: A => CustomClass[B]): CustomClass[B]
  def withFilter(p: A => Boolean): CustomClass[A]
  def foreach(b: A => Unit): Unit
}

object SimpleForComprehension {

  def main(args: Array[String]): Unit = {

    val names = Array("chris", "ed", "maurice")

    val capNames = for (e <- names) yield e.capitalize

    capNames.foreach(println)
    //Chris
    //Ed
    //Maurice

    println("-----------------------------------------------------")

    val userBase = List(User("Travis", 28),
      User("Kelly", 33),
      User("Jennifer", 44),
      User("Dennis", 23))

    val twentySomethings = for { //( is not used when definitions are present
      user <- userBase //Generators
      age = user.age //Definitions
      if age >= 20 && age < 30 //Filters
    }
      yield user.name  // i.e. add this to a list

    //Compiler converts it to following
    //val twentySomethings = userBase.
    // withFilter(((user) => user.age.$greater$eq(20).
    // $amp$amp(user.age.$less(30)))).
    // map(((user) => user.name));

    twentySomethings.foreach(name => println(name))
    // Travis
    // Dennis

    println("-----------------------------------------------------")

    val kvPair =
      for (i <- 1 to 10; //Every for comprehension begins with one or more generators
           j <- 1 until i)
        yield (i, j)

    //compiler generated
    val kvPair2 = 1.to(10).flatMap(((i) =>
      1.until(i).map(((j) =>
        scala.Tuple2(i, j)))));

    println(kvPair)
    // Vector((2,1), (3,1), (3,2), (4,1), (4,2), (4,3), (5,1), (5,2) ...
    println(kvPair2)
    // Vector((2,1), (3,1), (3,2), (4,1), (4,2), (4,3), (5,1), (5,2) ...

    val kvPair3 =
      for (i <- 1 to 10;
           j <- 1 until i
           if i % 2 == 0 && j % 2 == 0)
        yield (i, j)

    println(kvPair3)
    //Vector((4,2), (6,2), (6,4), (8,2), (8,4), (8,6), (10,2), (10,4), (10,6), (10,8))

    //Compiler converts it to following code
    val kvPair4 = 1.to(10).flatMap(((i) =>
      1.until(i).withFilter(((j) =>
        i.$percent(2).$eq$eq(0).$amp$amp(j.$percent(2).$eq$eq(0)))).map(((j) =>
        scala.Tuple2(i, j)))));


    println(kvPair4)


    println("-----------------------------------------------------")

    //Option works well with for expression, short circuiting the entire loop if there is a problem

    def makeInt(s: String): Option[Int] = {
      try {
        Some(s.trim.toInt)
      } catch {
        case e: Exception => None
      }
    }

    val res = for {
      i <- makeInt("1")
      j <- makeInt("2")
      k <- makeInt("3")
    } yield i + j + k

    //compiler generated, cleaned for easy reference
    val res1 = makeInt("1").flatMap(i =>
      makeInt("2").flatMap(j =>
        makeInt("3").map(k =>
          i.$plus(j).$plus(k)
        )
      )
    )

    println("res = " + res) //res = Some(6)
    println("res1 = " + res1) //res1 = Some(6)

    val res2 = for {
      i <- makeInt("1")
      j <- makeInt("t") //breaks the execution
      k <- makeInt("3")
    } yield i + j + k

    println("res2 = " + res2) //res2 = None

  }

}