-- phpMyAdmin SQL Dump
-- version 2.11.6
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Dec 06, 2022 at 12:02 PM
-- Server version: 5.0.51
-- PHP Version: 5.2.6

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `food_calorie`
--

-- --------------------------------------------------------

--
-- Table structure for table `admin`
--

CREATE TABLE `admin` (
  `username` varchar(20) NOT NULL,
  `password` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `admin`
--

INSERT INTO `admin` (`username`, `password`) VALUES
('admin', 'admin');

-- --------------------------------------------------------

--
-- Table structure for table `food`
--

CREATE TABLE `food` (
  `id` int(11) NOT NULL,
  `food` varchar(30) NOT NULL,
  `calorie` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `food`
--

INSERT INTO `food` (`id`, `food`, `calorie`) VALUES
(1, 'Apple', 52),
(2, 'Banana', 89),
(3, 'Brinjal', 25),
(4, 'Corn', 86),
(5, 'Carrot', 41),
(6, 'Dates', 282),
(7, 'Egg', 155),
(8, 'Mango', 60),
(9, 'Orange', 47),
(10, 'Pomegranate', 83),
(11, 'Potato', 77),
(12, 'Tomato', 18),
(13, 'Rice', 130),
(14, 'Fish', 206),
(15, 'Chicken', 239),
(16, 'Ghee', 900),
(17, 'Milk', 44),
(18, 'Noodles', 138),
(19, 'Parota', 335),
(20, 'Pizza', 266);

-- --------------------------------------------------------

--
-- Table structure for table `food_data`
--

CREATE TABLE `food_data` (
  `id` int(11) NOT NULL,
  `food` varchar(30) NOT NULL,
  `nutrient` text NOT NULL,
  `details` text NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `food_data`
--

INSERT INTO `food_data` (`id`, `food`, `nutrient`, `details`) VALUES
(1, 'Apple', 'Carbs: 28 grams, Fiber: 5 grams, Vitamin C: 10% of the Daily Value (DV), Copper: 6%, Potassium: 5%, Vitamin K: 4%', 'Apples are also a rich source of polyphenols, an important group of antioxidants. An increasing feeling of fullness works as a weight-loss strategy, as it helps manage your appetite.'),
(2, 'Banana', 'Potassium: 10%, Carbohydrate: 7%, Dietary fiber: 10%, Protein: 2%, Potassium: 10%, Vitamin C:14%, Iron: 1%, Vitamin B6: 20%, Magnesium: 6%', 'Bananas are a better source of energy than expensive sports drinks. Two bananas provide enough calories for an 1-1/2 hour workout or walk.'),
(3, 'Brinjal', 'Potassium 229mg: 6%, Dietary fiber 3g: 12%, Protein 1g: 2%, Vitamin C: 3%, Iron: 1%, Vitamin B6: 5%, Magnesium: 3%', 'Brinjal has antioxidants like vitamins A and C, which help protect your cells against damage. May Reduce the Risk of Heart Disease.May Promote Blood Sugar Control. May Have Cancer-Fighting Benefits. V'),
(4, 'Corn', 'Fat 1.2g: 1%, Potassium 270mg: 7%, Carbohydrate 19g: 6%, Dietary fiber 2.7g: 10%, Protein 3.2g: 6%, Vitamin C: 11%, Iron: 2%, Vitamin B6: 5%, Magnesium: 9%', 'Corn contains a heavy dose of sugar and carbohydrate. Taking too much corn can lead to weight gain. Yellow corn is a good source of the carotenoids lutein and zeaxanthin, which are good for eye health'),
(5, 'Carrot', 'Sodium 69mg: 2%, Potassium 320mg: 9%, Carbohydrate 10g: 3%, Dietary fiber 2.8g: 11%, Protein 0.9g: 1%, Vitamin C: 9%, Calcium: 3%, Iron: 1%, Vitamin B6: 5%, Magnesium: 3%', 'Rich source of dietary carotenoids. May support cholesterol balance and heart health. May help with weight loss goals. May reduce the risk of cancer. Improves Skin Health. Helps Improve Immunity.'),
(6, 'Dates', 'Potassium 656mg: 18%, Carbohydrate 75g: 25%, Dietary fiber 8g: 32%, Protein 2.5g: 5%, Calcium: 3%, Iron: 5%, Vitamin B6: 10%, Magnesium: 10%', 'Eating dates may help improve brain function. Dates is rich in fibre, so intake of 2-3 dates every day morning in an empty stomach can cure you from constipation and support the liver and protect it f'),
(7, 'Egg', 'Fat 3.3g: 16%, Cholesterol 373mg: 124%, Sodium 124mg: 5%, Potassium 126mg: 3%, Protein 13g: 26%, Calcium: 5%, Iron: 6%, Vitamin D: 21%, Vitamin B6: 5%, Cobalamin: 18%, Magnesium: 2%', 'Eggs raise good cholesterol. Eggs help maintain your eyesight. Get enough proteins and amino acids. Eggs Are Filling And Help With Weight Management. Eggs Are Among the Best Dietary Sources of Choline'),
(8, 'Mango', 'Potassium 168mg: 4%, Carbohydrate 15g: 5%, Protein 0.8g: 1%, Vitamin C: 60%, Calcium: 1%, Iron: 1%, Vitamin B6: 5%, Magnesium: 2%', 'May help prevent diabetes. Contains immune-boosting nutrients. Supports heart health. May improve digestive health. May support eye health. Green mangoes are rich in nutrients that promote collagen sy'),
(9, 'Orange', 'Potassium 181mg: 5%, Carbohydrate 12g: 4%, Dietary fiber 2.4g: 9%, Protein 0.9g: 1%, Vitamin C: 88%, Calcium: 4%, Vitamin B6: 5%, Magnesium: 2%', 'Protects your cells from damage. Helps your body make collagen, a protein that heals wounds and gives you smoother skin. Makes it easier to absorb iron to fight anemia. Boosts your immune system, your'),
(10, 'Pomegranate', 'Protein: 4.7g, Fat: 3.3g, Carbohydrates: 52g, Sugar: 38.6g, Fiber: 11.3g, Calcium: 28.2mg: 2% of the Daily Value, Iron: 0.85mg: 5%, Magnesium: 33.8mg: 8%, Phosphorus: 102mg, or 8%, Potassium 666mg: 13%, Vitamin C 28.8mg: 32%', 'Rich in antioxidants, protect the cells of your body from damage. Pomegranate fruit, juice, and oil can help kill cancer cells. The juice significantly reduced the frequency and severity of chest pain. May help reduce the formation of kidney stones.'),
(11, 'Potato', 'One medium potato delivers 610 mg of potassium, or roughly 17% of the daily value (DV). That''s about 40% more potassium than you''ll find in a banana, which only contains 422 mg, or 9% DV. In addition to potassium, 1 medium potato provides 5 grams of fiber, 4 grams of protein, vitamin C and magnesium.', 'Eating too many potatoes can present problems for blood sugar control in people with diabetes. May Improve Digestive Health. Naturally Gluten-Free. Rich in calcium and phosphorous, potatoes help strengthen bones.'),
(12, 'Tomato', 'Carbohydrates: 4.86g, Fat: 0.25g, Protein: 1.1g, Vitamin C: 17.1mg, 19% of the daily value, Potassium: 296mg, 6%, Water: 95%, Sugar: 2.6 grams. Fiber: 1.2 grams.', 'A tomato-rich diet has been linked to a reduced risk of heart disease. One whole tomato containing over four ounces of fluid and one and a half grams of fiber. The vitamin C in tomatoes acts as an antioxidant and is important for skin, bones.'),
(13, 'Rice', 'Rice is primarily composed of carbohydrate, which makes up almost 80% of its total dry weight. Most of the carbohydrate in rice is starch. Fat: 0.4g, Fiber: 0.6g, Protein: 4.4g, Vitamin B6: 5%, Magnesium: 3%', 'Rice is a rich source of carbohydrates, the body main fuel source. Rice is easy to digest and is has less saturated fats and has good cholesterol as compared to other foods.'),
(14, 'Fish', 'Fish is filled with omega-3 fatty acids and vitamins such as D and B2 (riboflavin). Fish is rich in calcium and phosphorus and a great source of minerals, such as iron, zinc, iodine, magnesium, and potassium. \r\n', 'Omega-3s play a role in sharpening memory. Fatty fish like salmon, herring, and mackerel have nutrients that may promote hair growth.'),
(15, 'Chicken', 'Protein: 24g, Fat: 3g, Niacin: 51%, Selenium: 36%, Phosphorus: 17%, Vitamin B6: 16%, Vitamin B12: 10%, Riboflavin: 9%, Zinc: 7%, Thiamine: 6%, Potassium: 5%, Copper: 4%', 'Research suggests that processed meat intake may be associated with a higher risk of heart disease, type 2 diabetes, and certain types of cancer. Reducing sodium intake has been shown to help decrease blood pressure levels. Fried and breaded chicken may be higher in unhealthy fats, carbs, and calories.'),
(16, 'Ghee', 'Water: 16%, Protein: 0.12 grams, Carbs: 0.01 grams, Sugar: 0.01 grams, Fiber: 0 grams, Fat: 11.52 grams. Saturated: 7.29 grams, Vitamin A: 107.5mcg, Vitamin E: 0.4mg, Vitamin K: 1.1mcg.', 'Ghee is almost completely pure fat, improving memory, increasing flexibility, and promoting healthy digestion. Maintaining healthy skin by locking in moisture. It also helps in reducing the liver load of cholesterol.'),
(17, 'Milk', 'Fat: 4.6g, Sodium: 95mg, Carbohydrates: 12g, Sugars: 12g, Protein: 8g, Calcium: 307mg 12%, Cholesterol 5 mg: 1%.', 'Milk Maintains Bone Density. Milk Promotes Muscle Growth. Milk Can Help Prevent Heartburn. Milk Can Help Fight Diseases. Milk Can Help Fight Depression. Milk Has Hair Hydrating Properties. Milk Helps Make Your Teeth Healthy.'),
(18, 'Noodles', 'Carbs: 27g, fat: 7g, Saturated fat: 3g, Protein: 4g, Fiber: 0.9g, Sodium: 861mg, Thiamine: 43% of the RDI, Cholesterol 29mg 9%, Carbohydrate 25g 8%, Iron 8%.', 'A vast majority of instant noodles are low in calories, but are also low in fibre and protein. They are also notorious for being high in fat, carbohydrates, and sodium. Insulin growth factors can trigger the hormones that produce acne.'),
(19, 'Parota', 'Fat 10g. 15%, Saturated Fat 4.6g. 23%, Sodium 357mg. 15%, Potassium 110mg. 3%, Carbohydrates 36g. 12%, Dietary Fiber 7.6g. 30%, Sugars 3.3g.', 'Whole wheat parotta delivers many important nutrients like fiber, minerals, protein, and antioxidants. Eating it amounts to depositing fats and cholesterol straight into your body. Parathas are considered unhealthy because most people stuff them.'),
(20, 'Pizza', 'Fat: 10.4g. 15%, Sodium: 640mg, Carbohydrates: 35.6g, Fiber: 2.5g, Sugars: 3.8g, Protein: 12.2g.', 'Pizza can present some decent nutrition when served up in the right form, but like most other foods, you can easily overdo this junk food. If you enjoy pizza but eat too much of it, you can face some ugly side effects of pizza that could lead to serious damage to your health.');

-- --------------------------------------------------------

--
-- Table structure for table `food_info`
--

CREATE TABLE `food_info` (
  `id` int(11) NOT NULL,
  `food` varchar(30) NOT NULL,
  `filename` varchar(100) NOT NULL,
  `calorie` varchar(30) NOT NULL,
  `weight` varchar(30) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `food_info`
--

INSERT INTO `food_info` (`id`, `food`, `filename`, `calorie`, `weight`) VALUES
(1, 'Apple', 'ap1.jpg', '33.8', '65'),
(2, 'Apple', 'ap10.jpg', '28.08', '54'),
(3, 'Apple', 'ap11.jpg', '24.96', '48'),
(4, 'Apple', 'ap12.jpg', '31.72', '61'),
(5, 'Apple', 'ap13.jpg', '32.76', '63'),
(6, 'Apple', 'ap14.jpg', '35.88', '69'),
(7, 'Apple', 'ap15.jpg', '21.84', '42'),
(8, 'Apple', 'ap16.jpg', '29.64', '57'),
(9, 'Apple', 'ap17.jpg', '30.16', '58'),
(10, 'Apple', 'ap18.jpg', '34.84', '67'),
(11, 'Apple', 'ap19.jpg', '34.32', '66'),
(12, 'Apple', 'ap2.jpg', '32.24', '62'),
(13, 'Apple', 'ap3.jpg', '31.72', '61'),
(14, 'Apple', 'ap4.jpg', '33.28', '64'),
(15, 'Apple', 'ap5.jpg', '36.4', '70'),
(16, 'Apple', 'ap6.jpg', '28.6', '55'),
(17, 'Apple', 'ap7.jpg', '30.16', '58'),
(18, 'Apple', 'ap8.jpg', '26.52', '51'),
(19, 'Apple', 'ap9.jpg', '27.56', '53'),
(20, 'Banana', 'bnn1.jpg', '31.15', '35'),
(21, 'Banana', 'bnn2.jpg', '40.05', '45'),
(22, 'Banana', 'bnn3.jpg', '37.38', '42'),
(23, 'Banana', 'bnn4.jpg', '34.71', '39'),
(24, 'Banana', 'bnn5.jpg', '32.93', '37'),
(25, 'Banana', 'bnn6.jpg', '38.27', '43'),
(26, 'Banana', 'bnn7.jpg', '41.83', '47'),
(27, 'Banana', 'bnn8.jpg', '45.39', '51'),
(28, 'Brinjal', 'brj1.jpg', '13.0', '52'),
(29, 'Brinjal', 'brj2.jpg', '10.5', '42'),
(30, 'Brinjal', 'brj3.jpg', '11.25', '45'),
(31, 'Brinjal', 'brj4.jpg', '7.75', '31'),
(32, 'Brinjal', 'brj5.jpg', '9.0', '36'),
(33, 'Brinjal', 'brj6.jpg', '7.25', '29'),
(34, 'Brinjal', 'brj7.jpg', '10.75', '43'),
(35, 'Corn', 'cn1.jpg', '76.54', '89'),
(36, 'Corn', 'cn2.jpg', '90.3', '105'),
(37, 'Corn', 'cn3.jpg', '115.24', '134'),
(38, 'Corn', 'cn4.jpg', '124.7', '145'),
(39, 'Corn', 'cn5.jpg', '134.16', '156'),
(40, 'Corn', 'cn6.jpg', '126.42', '147'),
(41, 'Corn', 'cn7.jpg', '156.52', '182'),
(42, 'Corn', 'cn8.jpg', '125.56', '146'),
(43, 'Carrot', 'crrt1.jpg', '21.32', '52'),
(44, 'Carrot', 'crrt10.jpg', '15.58', '38'),
(45, 'Carrot', 'crrt2.jpg', '18.04', '44'),
(46, 'Carrot', 'crrt3.jpg', '20.09', '49'),
(47, 'Carrot', 'crrt4.jpg', '20.91', '51'),
(48, 'Carrot', 'crrt5.jpg', '22.96', '56'),
(49, 'Carrot', 'crrt6.jpg', '21.73', '53'),
(50, 'Carrot', 'crrt7.jpg', '23.37', '57'),
(51, 'Carrot', 'crrt8.jpg', '19.68', '48'),
(52, 'Carrot', 'crrt9.jpeg', '23.78', '58'),
(53, 'Dates', 'dt1.jpg', '236.88', '84'),
(54, 'Dates', 'dt10.jpg', '205.86', '73'),
(55, 'Dates', 'dt11.jpg', '194.58', '69'),
(56, 'Dates', 'dt2.jpg', '217.14', '77'),
(57, 'Dates', 'dt3.jpg', '174.84', '62'),
(58, 'Dates', 'dt4.jpg', '177.66', '63'),
(59, 'Dates', 'dt5.jpg', '188.94', '67'),
(60, 'Dates', 'dt6.jpg', '200.22', '71'),
(61, 'Dates', 'dt7.jpg', '239.7', '85'),
(62, 'Dates', 'dt8.jpg', '267.9', '95'),
(63, 'Dates', 'dt9.jpg', '70.5', '25'),
(64, 'Egg', 'eg1.jpg', '68.2', '44'),
(65, 'Egg', 'eg2.jpg', '100.75', '65'),
(66, 'Egg', 'eg3.jpg', '114.7', '74'),
(67, 'Egg', 'eg4.jpg', '52.7', '34'),
(68, 'Egg', 'eg5.jpg', '125.55', '81'),
(69, 'Egg', 'eg6.jpg', '105.4', '68'),
(70, 'Egg', 'eg7.jpg', '88.35', '57'),
(71, 'Egg', 'eg8.jpg', '75.95', '49'),
(72, 'Mango', 'mng1.jpg', '43.2', '72'),
(73, 'Mango', 'mng10.jpg', '21.6', '36'),
(74, 'Mango', 'mng2.jpg', '30.6', '51'),
(75, 'Mango', 'mng3.jpg', '27.0', '45'),
(76, 'Mango', 'mng4.jpg', '54.0', '90'),
(77, 'Mango', 'mng5.jpg', '52.2', '87'),
(78, 'Mango', 'mng6.jpg', '57.6', '96'),
(79, 'Mango', 'mng7.jpg', '44.4', '74'),
(80, 'Mango', 'mng8.jpg', '51.0', '85'),
(81, 'Mango', 'mng9.jpg', '39.6', '66'),
(82, 'Orange', 'ong1.jpg', '21.15', '45'),
(83, 'Orange', 'ong2.jpg', '31.02', '66'),
(84, 'Orange', 'ong3.jpg', '33.84', '72'),
(85, 'Orange', 'ong4.jpg', '25.38', '54'),
(86, 'Orange', 'ong5.jpg', '28.67', '61'),
(87, 'Orange', 'ong6.jpg', '34.78', '74'),
(88, 'Orange', 'ong7.jpg', '24.91', '53'),
(89, 'Orange', 'ong8.jpg', '27.26', '58'),
(90, 'Orange', 'ong9.jpg', '25.38', '54'),
(91, 'Pomegranate', 'pmg1.jpg', '107.9', '130'),
(92, 'Pomegranate', 'pmg10.jpg', '58.1', '70'),
(93, 'Pomegranate', 'pmg2.jpg', '53.95', '65'),
(94, 'Pomegranate', 'pmg3.jpg', '49.8', '60'),
(95, 'Pomegranate', 'pmg4.jpg', '74.7', '90'),
(96, 'Pomegranate', 'pmg5.jpg', '99.6', '120'),
(97, 'Pomegranate', 'pmg6.jpg', '33.2', '40'),
(98, 'Pomegranate', 'pmg7.jpg', '48.14', '58'),
(99, 'Pomegranate', 'pmg8.jpg', '59.76', '72'),
(100, 'Pomegranate', 'pmg9.jpg', '79.68', '96'),
(101, 'Potato', 'pto1.jpg', '65.45', '85'),
(102, 'Potato', 'pto2.jpg', '118.58', '154'),
(103, 'Potato', 'pto3.jpg', '53.13', '69'),
(104, 'Potato', 'pto4.jpg', '43.89', '57'),
(105, 'Potato', 'pto5.jpg', '120.12', '156'),
(106, 'Potato', 'pto6.jpg', '68.53', '89'),
(107, 'Potato', 'pto7.jpg', '56.98', '74'),
(108, 'Potato', 'pto8.jpg', '38.5', '50'),
(109, 'Tomato', 'tm1.jpg', '13.5', '75'),
(110, 'Tomato', 'tm10.jpg', '21.6', '120'),
(111, 'Tomato', 'tm11.jpg', '6.48', '36'),
(112, 'Tomato', 'tm12.jpg', '17.1', '95'),
(113, 'Tomato', 'tm2.jpg', '18.0', '100'),
(114, 'Tomato', 'tm3.jpg', '13.32', '74'),
(115, 'Tomato', 'tm4.jpg', '26.64', '148'),
(116, 'Tomato', 'tm5.jpg', '35.28', '196'),
(117, 'Tomato', 'tm6.jpg', '25.2', '140'),
(118, 'Tomato', 'tm7.jpg', '5.4', '30'),
(119, 'Tomato', 'tm8.jpg', '4.5', '25'),
(120, 'Tomato', 'tm9.jpg', '8.46', '47'),
(121, 'Rice', 'f1210.jpg', '130.0', '100'),
(122, 'Rice', 'f1221.jpg', '195.0', '150'),
(123, 'Rice', 'f1232.jpg', '260.0', '200'),
(124, 'Rice', 'f1243.jpg', '136.5', '105'),
(125, 'Rice', 'f12510.jpg', '260.0', '200'),
(126, 'Rice', 'f12611_1.jpg', '195.0', '150'),
(127, 'Rice', 'f12711.jpg', '260.0', '200'),
(128, 'Fish', 'f12860973839.jpg', '309.0', '150'),
(129, 'Fish', 'f1291664308524_new-project-57.jpg', '251.32', '122'),
(130, 'Fish', 'f130bcgh6jg.jpg', '144.2', '70'),
(131, 'Fish', 'f131Chettinad-fish-fry-1B.jpg', '164.8', '80'),
(132, 'Fish', 'f132fdfd56hg.jpg', '329.6', '160'),
(133, 'Fish', 'f133fish-fry-masala.jpg', '257.5', '125'),
(134, 'Fish', 'f134fish-lead-story-.jpg', '103.0', '50'),
(135, 'Fish', 'f135ftytu657uy.jpg', '61.8', '30'),
(136, 'Fish', 'f136vanjaram-fish-fry.jpg', '72.1', '35'),
(137, 'Chicken', 'f137960xfd.jpg', '1434.0', '600'),
(138, 'Chicken', 'f138df43.jpg', '262.9', '110'),
(139, 'Chicken', 'f139dsds4re43.jpg', '310.7', '130'),
(140, 'Chicken', 'f140nrty54fg.jpg', '107.55', '45'),
(141, 'Chicken', 'f141vm56yu.jpg', '478.0', '200'),
(142, 'Ghee', 'f14245thhjf.jpg', '1620.0', '180'),
(143, 'Ghee', 'f143720_404_1-1.jpg', '900.0', '100'),
(144, 'Ghee', 'f1442000x600.jpg', '270.0', '30'),
(145, 'Ghee', 'f145a2-organic-ghee.jpg', '405.0', '45'),
(146, 'Ghee', 'f146dairy-farm-ghee-500x500.jpg', '720.0', '80'),
(147, 'Ghee', 'f147Fresh-Desi-Ghee.jpg', '225.0', '25'),
(148, 'Milk', 'f148hfdr5gf.jpg', '66.0', '150'),
(149, 'Milk', 'f149kefjhvsfd934.jpg', '88.0', '200'),
(150, 'Milk', 'f150fggf78hy.jpg', '96.8', '220'),
(151, 'Milk', 'f151Milk-Benefits.jpg', '39.6', '90'),
(152, 'Milk', 'f152milk.jpg', '46.2', '105'),
(153, 'Noodles', 'f15310.jpg', '207.0', '150'),
(154, 'Noodles', 'f154102_1.jpg', '248.4', '180'),
(155, 'Noodles', 'f1550.jpg', '96.6', '70'),
(156, 'Noodles', 'f156104.jpg', '124.2', '90'),
(157, 'Noodles', 'f157113.jpg', '110.4', '80'),
(158, 'Parota', 'f158376298-malabar-parotta_1.jpg', '1005.0', '300'),
(159, 'Parota', 'f159382419-parota.jpg', '837.5', '250'),
(160, 'Parota', 'f160xxcfdw.jpg', '536.0', '160'),
(161, 'Parota', 'f161gngyh4.jpg', '603.0', '180'),
(162, 'Parota', 'f162fdtre56ed.jpg', '871.0', '260'),
(163, 'Pizza', 'f163001.jpg', '478.8', '180'),
(164, 'Pizza', 'f164005.jpg', '558.6', '210'),
(165, 'Pizza', 'f165007.jpg', '611.8', '230'),
(166, 'Pizza', 'f166013.jpg', '452.2', '170'),
(167, 'Pizza', 'f167020.jpg', '505.4', '190');

-- --------------------------------------------------------

--
-- Table structure for table `history`
--

CREATE TABLE `history` (
  `id` int(11) NOT NULL,
  `uname` varchar(20) NOT NULL,
  `food` varchar(30) NOT NULL,
  `filename` varchar(100) NOT NULL,
  `calorie` varchar(20) NOT NULL,
  `gram` varchar(20) NOT NULL,
  `date_time` timestamp NOT NULL default CURRENT_TIMESTAMP on update CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `history`
--

INSERT INTO `history` (`id`, `uname`, `food`, `filename`, `calorie`, `gram`, `date_time`) VALUES
(1, 'raj', 'Brinjal', 'brj7.jpg', '25', '100g', '2022-12-06 07:47:11'),
(2, 'raj', 'Mango', 'mng9.jpg', '60', '100g', '2022-12-06 10:56:03'),
(3, 'usha', 'Banana', 'bnn7.jpg', '89', '100g', '2022-12-06 13:22:19'),
(4, 'usha', 'Egg', 'eg6.jpg', '155', '100g', '2022-12-06 13:23:41');

-- --------------------------------------------------------

--
-- Table structure for table `register`
--

CREATE TABLE `register` (
  `id` int(11) NOT NULL,
  `name` varchar(20) NOT NULL,
  `mobile` bigint(20) NOT NULL,
  `email` varchar(40) NOT NULL,
  `uname` varchar(20) NOT NULL,
  `pass` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `register`
--

INSERT INTO `register` (`id`, `name`, `mobile`, `email`, `uname`, `pass`) VALUES
(1, 'Raj', 9638527412, 'raj@gmail.com', 'raj', '1234'),
(2, 'Usha', 9875612525, 'usha@gmail.com', 'usha', '1234');
